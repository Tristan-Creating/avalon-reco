"""Shared fixtures: a tiny deterministic university for fast tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avalon_reco.model import CourseRecommender  # noqa: E402


@pytest.fixture(scope="session")
def tiny_university():
    """2 faculties × 2 levels, 8 courses, 6 students with clear taste clusters.

    Students 1-3 are SCI bachelors who all take courses 101+102; students 4-6
    are LAW bachelors around courses 201+202. Course 103/203 are the 'next'
    courses partially taken — the CF signal to recover. 301/302 are master
    courses (level filter targets). Student 7 exists in metadata only
    (cold start).
    """
    courses = pd.DataFrame([
        (101, "SCI-101", "Algorithms", "Science", "bachelor", 5),
        (102, "SCI-102", "Databases", "Science", "bachelor", 5),
        (103, "SCI-103", "Statistics", "Science", "bachelor", 5),
        (201, "LAW-201", "Contracts", "Law", "bachelor", 5),
        (202, "LAW-202", "Torts", "Law", "bachelor", 5),
        (203, "LAW-203", "EU Law", "Law", "bachelor", 5),
        (301, "SCI-301", "Advanced ML", "Science", "master", 5),
        (302, "LAW-301", "Tax Law", "Law", "master", 5),
    ], columns=["course_id", "code", "title", "faculty", "level", "ects"])

    students = pd.DataFrame([
        (1, "Science", "bachelor"), (2, "Science", "bachelor"), (3, "Science", "bachelor"),
        (4, "Law", "bachelor"), (5, "Law", "bachelor"), (6, "Law", "bachelor"),
        (7, "Law", "bachelor"),  # no interactions: cold start
    ], columns=["student_id", "faculty", "level"])

    rows = [
        (1, 101, 2024), (1, 102, 2024), (1, 103, 2024),
        (2, 101, 2024), (2, 102, 2024), (2, 103, 2024),
        (3, 101, 2024), (3, 102, 2024),          # 3 hasn't taken 103 yet
        (4, 201, 2024), (4, 202, 2024), (4, 203, 2024),
        (5, 201, 2024), (5, 202, 2024), (5, 203, 2024),
        (6, 201, 2024), (6, 202, 2024),          # 6 hasn't taken 203 yet
    ]
    interactions = pd.DataFrame(rows, columns=["student_id", "course_id", "academic_year"])
    interactions["status"] = "completed"
    return interactions, students, courses


@pytest.fixture(scope="session")
def model(tiny_university):
    interactions, students, courses = tiny_university
    return CourseRecommender.fit(interactions, students, courses, top_n_neighbors=8)
