#!/usr/bin/env python3
"""Data drift monitoring with Evidently.

Compares the live enrollment stream (recent window from the Block 3 landing
zone) against the training reference (historical enrollments in the
warehouse) on the features the recommender actually depends on: faculty mix,
level mix, courses per student.

Outputs:
    monitoring/reports/drift-<stamp>.html    full Evidently report (for humans)
    ops.ml_drift row in Postgres             share of drifted features + flag
                                             (for Grafana, next to pipeline runs)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DSN = os.getenv(
    "AVALON_DSN", "postgresql://avalon:avalon_dev_password@localhost:5432/avalon"
)

REFERENCE_QUERY = """
SELECT s.faculty, s.program_level AS level, c.faculty AS course_faculty,
       count(*) OVER (PARTITION BY f.student_key) AS courses_per_student
FROM warehouse.fact_enrollments f
JOIN warehouse.dim_student s ON s.student_key = f.student_key
JOIN warehouse.dim_course  c ON c.course_key = f.course_key
WHERE f.record_source = 'batch'
"""

CURRENT_QUERY = """
SELECT s.faculty, s.program_level AS level, c.faculty AS course_faculty,
       count(*) OVER (PARTITION BY e.student_id) AS courses_per_student
FROM staging.raw_enrollment_events e
JOIN warehouse.dim_student s ON s.student_key = e.student_id
JOIN warehouse.dim_course  c ON c.course_key = e.course_id
WHERE e.loaded_at > now() - interval %(window)s
"""

OPS_DDL = """
CREATE SCHEMA IF NOT EXISTS ops;
CREATE TABLE IF NOT EXISTS ops.ml_drift (
    checked_at      TIMESTAMPTZ PRIMARY KEY DEFAULT now(),
    window_hours    INT NOT NULL,
    n_reference     BIGINT NOT NULL,
    n_current       BIGINT NOT NULL,
    drift_share     NUMERIC(5, 4) NOT NULL,
    dataset_drift   BOOLEAN NOT NULL
);
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--reports-dir", type=Path, default=Path(__file__).parent / "reports")
    args = parser.parse_args()

    import psycopg2
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    with psycopg2.connect(args.dsn) as conn:
        reference = pd.read_sql(REFERENCE_QUERY, conn)
        current = pd.read_sql(
            CURRENT_QUERY, conn, params={"window": f"{args.window_hours} hours"}
        )

    if current.empty:
        print("no streamed enrollments in the window — nothing to compare")
        return 0

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    html_path = args.reports_dir / f"drift-{stamp}.html"
    report.save_html(str(html_path))

    summary = report.as_dict()["metrics"][0]["result"]
    drift_share = summary["share_of_drifted_columns"]
    dataset_drift = summary["dataset_drift"]

    with psycopg2.connect(args.dsn) as conn, conn.cursor() as cur:
        cur.execute(OPS_DDL)
        cur.execute(
            "INSERT INTO ops.ml_drift (window_hours, n_reference, n_current, drift_share, dataset_drift)"
            " VALUES (%s, %s, %s, %s, %s)",
            (args.window_hours, len(reference), len(current), drift_share, dataset_drift),
        )

    print(f"report: {html_path}")
    print(f"drifted feature share: {drift_share:.0%} — dataset drift: {dataset_drift}")
    return 1 if dataset_drift else 0


if __name__ == "__main__":
    sys.exit(main())
