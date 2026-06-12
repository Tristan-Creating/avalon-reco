#!/usr/bin/env python3
"""Seed a miniature warehouse for the CI training smoke test.

Creates just the three warehouse tables the trainer reads and fills them with
a deterministic miniature university: enough structure for collaborative
filtering to find signal (faculty clusters across two academic years).
"""

import os
import random

import psycopg2

DSN = os.getenv("AVALON_DSN", "postgresql://avalon:avalon_dev_password@localhost:5432/avalon")

DDL = """
CREATE SCHEMA IF NOT EXISTS warehouse;
DROP TABLE IF EXISTS warehouse.fact_enrollments, warehouse.dim_student, warehouse.dim_course;
CREATE TABLE warehouse.dim_student (
    student_key INT PRIMARY KEY, student_id INT, full_name TEXT,
    program TEXT, program_level TEXT, faculty TEXT, enrollment_year INT, status TEXT
);
CREATE TABLE warehouse.dim_course (
    course_key INT PRIMARY KEY, course_id INT, code TEXT, title TEXT,
    faculty TEXT, level TEXT, semester TEXT, ects INT, teacher TEXT
);
CREATE TABLE warehouse.fact_enrollments (
    enrollment_sk TEXT PRIMARY KEY, date_key INT, student_key INT, course_key INT,
    academic_year INT, grade NUMERIC, ects_attempted INT, ects_earned INT,
    status TEXT, record_source TEXT
);
"""

FACULTIES = ["Science", "Law", "Business"]


def main() -> None:
    rng = random.Random(7)
    students = [(i, f"Student {i}", FACULTIES[i % 3], "bachelor") for i in range(1, 121)]
    courses = [
        (cid, f"{fac[:3].upper()}-{cid}", f"Course {cid}", fac, "bachelor")
        for cid, fac in ((100 + i, FACULTIES[i % 3]) for i in range(30))
    ]
    by_faculty: dict = {}
    for cid, _, _, fac, _ in courses:
        by_faculty.setdefault(fac, []).append(cid)

    facts = []
    for sid, _, fac, _ in students:
        for year in (2024, 2025):
            own = rng.sample(by_faculty[fac], 5)
            other = rng.sample([c for f, ids in by_faculty.items() if f != fac for c in ids], 1)
            for cid in own + other:
                facts.append((f"{sid}:{cid}:{year}", year * 10000 + 901, sid, cid, year,
                              None, 5, 0, "in_progress", "batch"))

    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.executemany(
            "INSERT INTO warehouse.dim_student VALUES (%s, %s, %s, 'CI Program', %s, %s, 2024, 'active')",
            [(sid, sid, name, lvl, fac) for sid, name, fac, lvl in students],
        )
        cur.executemany(
            "INSERT INTO warehouse.dim_course VALUES (%s, %s, %s, %s, %s, %s, 'S1', %s, 'CI Teacher')",
            [(cid, cid, code, title, fac, lvl, 5) for cid, code, title, fac, lvl in courses],
        )
        cur.executemany(
            "INSERT INTO warehouse.fact_enrollments VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT DO NOTHING",
            facts,
        )
    print(f"seeded {len(students)} students, {len(courses)} courses, {len(facts)} enrollments")


if __name__ == "__main__":
    main()
