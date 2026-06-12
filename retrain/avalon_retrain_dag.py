"""Weekly retraining DAG — drop into the Block 3 Airflow dags/ folder.

The retrain script applies its own champion/challenger gate; a non-promotion
(exit code 2) is surfaced as a *skipped* promotion, not a failure.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="avalon_reco_retrain",
    description="Weekly course-recommender retraining with promotion gate",
    schedule="0 3 * * 1",  # Mondays 03:00 UTC
    start_date=pendulum.datetime(2026, 6, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ml-platform", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["avalon", "ml", "retraining"],
) as dag:

    retrain = BashOperator(
        task_id="retrain_and_gate",
        bash_command=(
            "python /opt/airflow/reco/retrain/retrain.py "
            "--models-dir /opt/airflow/reco/models "
            "--api-url http://reco-api:8000 "
            "|| [ $? -eq 2 ]  # exit 2 = quality gate held the champion, not an error"
        ),
        append_env=True,
        env={"AVALON_DSN": "postgresql://avalon:avalon_dev_password@postgres:5432/avalon"},
    )
