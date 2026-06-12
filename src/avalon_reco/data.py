"""Data access for the recommender: warehouse → interaction frames.

The model trains on the star schema built by Block 3 (dbt), so streamed
enrollments are automatically part of the next training set.
"""

from __future__ import annotations

import pandas as pd

INTERACTIONS_QUERY = """
SELECT
    f.student_key   AS student_id,
    f.course_key    AS course_id,
    f.academic_year,
    f.status
FROM warehouse.fact_enrollments f
WHERE f.status <> 'dropped'      -- a dropped course is not a positive signal
"""

STUDENTS_QUERY = """
SELECT student_key AS student_id, faculty, program_level AS level
FROM warehouse.dim_student
"""

COURSES_QUERY = """
SELECT course_key AS course_id, code, title, faculty, level, ects
FROM warehouse.dim_course
"""


def load_frames(dsn: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (interactions, students, courses) from the warehouse."""
    import psycopg2

    with psycopg2.connect(dsn) as conn:
        interactions = pd.read_sql(INTERACTIONS_QUERY, conn)
        students = pd.read_sql(STUDENTS_QUERY, conn)
        courses = pd.read_sql(COURSES_QUERY, conn)
    return interactions, students, courses


def temporal_split(
    interactions: pd.DataFrame, test_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on everything before `test_year`, test on `test_year` itself.

    A temporal split mirrors production reality (predict next year's
    enrollments from history) and avoids the leakage a random split would
    introduce.
    """
    train = interactions[interactions["academic_year"] < test_year]
    test = interactions[interactions["academic_year"] == test_year]
    return train, test
