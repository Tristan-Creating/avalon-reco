"""Model behavior tests on the tiny deterministic university."""

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avalon_reco.model import CourseRecommender  # noqa: E402


class TestCollaborativeFiltering:
    def test_recovers_cluster_signal(self, model):
        """Student 3 took 101+102; everyone like them also took 103."""
        recs = model.recommend(3, k=3)
        assert recs[0].course_id == 103
        assert recs[0].strategy == "collaborative"

    def test_cross_faculty_signal(self, model):
        """Student 6 (Law) should be pointed at the missing Law course."""
        recs = model.recommend(6, k=3)
        assert recs[0].course_id == 203

    def test_never_recommends_taken_courses(self, model):
        recs = model.recommend(1, k=8)
        taken = {101, 102, 103}
        assert taken.isdisjoint({r.course_id for r in recs})

    def test_level_filter(self, model):
        """Bachelor students never see master courses."""
        for student_id in (1, 3, 6):
            for r in model.recommend(student_id, k=8):
                assert r.level == "bachelor"

    def test_k_respected(self, model):
        assert len(model.recommend(3, k=2)) == 2


class TestColdStart:
    def test_known_student_without_history(self, model):
        """Student 7 is in dim_student but has no enrollments."""
        recs = model.recommend(7, k=3)
        assert recs, "cold start must still produce recommendations"
        assert recs[0].strategy == "popular_faculty"
        assert all(r.faculty == "Law" for r in recs)

    def test_unknown_student_with_context(self, model):
        recs = model.recommend(999, k=3, student_info=("Science", "bachelor"))
        assert recs[0].strategy == "popular_faculty"
        assert all(r.faculty == "Science" for r in recs)

    def test_unknown_student_without_context(self, model):
        recs = model.recommend(999, k=3)
        assert recs, "must degrade to level-wide popularity, not fail"
        assert recs[0].strategy == "popular_level"


class TestArtifactRoundtrip:
    def test_joblib_roundtrip(self, model, tmp_path):
        path = tmp_path / "model.joblib"
        joblib.dump(model.to_artifact(), path)
        loaded = CourseRecommender.from_artifact(joblib.load(path))
        original = [r.course_id for r in model.recommend(3, k=5)]
        roundtripped = [r.course_id for r in loaded.recommend(3, k=5)]
        assert original == roundtripped

    def test_bad_format_rejected(self, model):
        artifact = model.to_artifact()
        artifact["format"] = 99
        try:
            CourseRecommender.from_artifact(artifact)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
