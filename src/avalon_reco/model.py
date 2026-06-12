"""Course recommender: item-item collaborative filtering on implicit feedback.

Why this model:
- Enrollments are implicit positive signals (no ratings) — item-item cosine
  similarity on the student × course matrix is the standard, robust baseline
  for this regime, trains in seconds at Avalon's scale, and its
  recommendations are explainable ("because you took X").
- Domain constraint baked in: a student is only ever recommended courses of
  their own academic level (bachelor/master), never a course already taken.
- Tiered cold start: students without history get the most-enrolled courses
  of their faculty + level; unknown faculty falls back to level-wide
  popularity.

The trained artifact is a plain dict (joblib) — no custom classes inside, so
the API can load it without importing training code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse

ARTIFACT_FORMAT = 2


@dataclass
class Recommendation:
    course_id: int
    score: float
    code: str
    title: str
    faculty: str
    level: str
    strategy: str  # "collaborative" | "popular_faculty" | "popular_level"


@dataclass
class CourseRecommender:
    """Wraps the artifact dict with the recommendation logic."""

    item_sim: sparse.csr_matrix                 # courses × courses, top-N sparsified
    course_ids: np.ndarray                      # column index → course_id
    course_pos: dict[int, int]                  # course_id → column index
    histories: dict[int, list[int]]             # student_id → [course_id]
    student_meta: dict[int, tuple[str, str]]    # student_id → (faculty, level)
    course_meta: pd.DataFrame                   # indexed by course_id
    popular_by_faculty_level: dict[tuple[str, str], list[int]]
    popular_by_level: dict[str, list[int]]
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ---------- training ----------

    @classmethod
    def fit(
        cls,
        interactions: pd.DataFrame,
        students: pd.DataFrame,
        courses: pd.DataFrame,
        top_n_neighbors: int = 50,
    ) -> "CourseRecommender":
        course_ids = np.sort(courses["course_id"].unique())
        course_pos = {cid: i for i, cid in enumerate(course_ids)}
        student_ids = np.sort(interactions["student_id"].unique())
        student_pos = {sid: i for i, sid in enumerate(student_ids)}

        rows = interactions["student_id"].map(student_pos).to_numpy()
        cols = interactions["course_id"].map(course_pos).to_numpy()
        data = np.ones(len(interactions), dtype=np.float32)
        matrix = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(student_ids), len(course_ids))
        )
        matrix.data[:] = 1.0  # collapse duplicate enrollments

        # cosine similarity between course columns
        co = (matrix.T @ matrix).toarray().astype(np.float32)
        norms = np.sqrt(np.diag(co))
        norms[norms == 0] = 1.0
        sim = co / np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)

        # keep only the strongest neighbors per course (noise + size control)
        if top_n_neighbors < sim.shape[0]:
            for i in range(sim.shape[0]):
                row = sim[i]
                cutoff = np.partition(row, -top_n_neighbors)[-top_n_neighbors]
                row[row < cutoff] = 0.0

        histories = (
            interactions.groupby("student_id")["course_id"].agg(lambda s: sorted(set(s))).to_dict()
        )
        student_meta = {
            int(r.student_id): (r.faculty, r.level) for r in students.itertuples()
        }
        course_meta = courses.set_index("course_id")

        enriched = interactions.merge(courses, on="course_id")
        pop_fl = {
            (fac, lvl): grp["course_id"].value_counts().index.tolist()
            for (fac, lvl), grp in enriched.groupby(["faculty", "level"])
        }
        pop_l = {
            lvl: grp["course_id"].value_counts().index.tolist()
            for lvl, grp in enriched.groupby("level")
        }

        return cls(
            item_sim=sparse.csr_matrix(sim),
            course_ids=course_ids,
            course_pos=course_pos,
            histories=histories,
            student_meta=student_meta,
            course_meta=course_meta,
            popular_by_faculty_level=pop_fl,
            popular_by_level=pop_l,
        )

    # ---------- inference ----------

    def _to_recommendations(self, course_ids, scores, strategy) -> list[Recommendation]:
        out = []
        for cid, score in zip(course_ids, scores):
            meta = self.course_meta.loc[cid]
            out.append(Recommendation(
                course_id=int(cid), score=round(float(score), 4),
                code=meta["code"], title=meta["title"], faculty=meta["faculty"],
                level=meta["level"], strategy=strategy,
            ))
        return out

    def _popular(self, k: int, level: str, faculty: Optional[str], exclude: set[int]) -> list[Recommendation]:
        if faculty and (faculty, level) in self.popular_by_faculty_level:
            pool, strategy = self.popular_by_faculty_level[(faculty, level)], "popular_faculty"
        else:
            pool, strategy = self.popular_by_level.get(level, []), "popular_level"
        picked = [cid for cid in pool if cid not in exclude][:k]
        if len(picked) < k:  # faculty pool exhausted → widen to the whole level
            wider = self.popular_by_level.get(level, [])
            picked += [c for c in wider if c not in exclude and c not in picked][: k - len(picked)]
        scores = np.linspace(1.0, 0.5, num=len(picked)) if picked else []
        return self._to_recommendations(picked, scores, strategy)

    def recommend(
        self,
        student_id: int,
        k: int = 10,
        student_info: Optional[tuple[str, str]] = None,
    ) -> list[Recommendation]:
        """Top-k course recommendations for a student.

        `student_info` (faculty, level) lets the caller supply context for
        students unseen at training time (the API looks them up in the
        warehouse); without it, unknown students get level-wide popularity
        for the most common level.
        """
        info = self.student_meta.get(student_id) or student_info
        faculty, level = info if info else (None, None)
        history = self.histories.get(student_id, [])
        seen = set(history)

        if not history:
            return self._popular(k, level or "bachelor", faculty, seen)

        positions = [self.course_pos[c] for c in history if c in self.course_pos]
        scores = np.asarray(self.item_sim[positions].sum(axis=0)).ravel()

        # never recommend a taken course; enforce level match when known
        for pos in positions:
            scores[pos] = -np.inf
        if level is not None:
            level_ok = self.course_meta.loc[self.course_ids, "level"].to_numpy() == level
            scores[~level_ok] = -np.inf

        order = np.argsort(-scores)[:k]
        order = order[np.isfinite(scores[order]) & (scores[order] > 0)]
        recs = self._to_recommendations(self.course_ids[order], scores[order], "collaborative")

        if len(recs) < k:  # thin history → top up with popularity
            have = {r.course_id for r in recs} | seen
            recs += self._popular(k - len(recs), level or "bachelor", faculty, have)
        return recs[:k]

    # ---------- persistence ----------

    def to_artifact(self) -> dict:
        return {
            "format": ARTIFACT_FORMAT,
            "trained_at": self.trained_at,
            "item_sim": self.item_sim,
            "course_ids": self.course_ids,
            "course_pos": self.course_pos,
            "histories": self.histories,
            "student_meta": self.student_meta,
            "course_meta": self.course_meta,
            "popular_by_faculty_level": self.popular_by_faculty_level,
            "popular_by_level": self.popular_by_level,
        }

    @classmethod
    def from_artifact(cls, artifact: dict) -> "CourseRecommender":
        if artifact.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"unsupported artifact format: {artifact.get('format')}")
        fields = {k: v for k, v in artifact.items() if k != "format"}
        return cls(**fields)
