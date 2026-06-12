#!/usr/bin/env python3
"""Automated retraining with a champion/challenger promotion gate.

Flow (run by cron, Airflow, or manually):
    1. retrain on the current warehouse (which now includes streamed
       enrollments from the Block 3 pipeline),
    2. evaluate challenger on the temporal hold-out,
    3. promote only if precision@k does not regress beyond --tolerance
       against the current champion,
    4. on promotion: update models/latest.joblib and hot-reload the API.

Exit codes: 0 promoted / 2 not promoted (quality gate) — automation can act
on them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avalon_reco.train import DEFAULT_DSN, save, train_and_evaluate  # noqa: E402


def current_champion_precision(models_dir: Path) -> float | None:
    registry_path = models_dir / "registry.json"
    if not registry_path.exists():
        return None
    promoted = [e for e in json.loads(registry_path.read_text()) if e.get("promoted")]
    if not promoted:
        return None
    return promoted[-1]["model_metrics"]["precision_at_k"] if "model_metrics" in promoted[-1] \
        else promoted[-1]["model"]["precision_at_k"]


def reload_api(api_url: str, token: str) -> bool:
    req = urllib.request.Request(f"{api_url}/admin/reload", method="POST")
    req.add_header("X-Admin-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print("API reload:", resp.read().decode())
        return True
    except Exception as exc:
        print(f"API reload failed (non-fatal): {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="max precision@k regression allowed for promotion")
    parser.add_argument("--api-url", default=os.getenv("RECO_API_URL", ""))
    parser.add_argument("--admin-token", default=os.getenv("ADMIN_TOKEN", "avalon_admin"))
    args = parser.parse_args()

    champion = current_champion_precision(args.models_dir)
    model, report = train_and_evaluate(args.dsn, args.test_year, args.k)
    challenger = report["model"]["precision_at_k"]

    if champion is None:
        promote, reason = True, "no champion yet"
    elif challenger >= champion - args.tolerance:
        promote, reason = True, f"challenger {challenger} vs champion {champion}"
    else:
        promote, reason = False, f"regression: challenger {challenger} < champion {champion} - {args.tolerance}"

    print(f"promotion decision: {promote} ({reason})")
    path = save(model, report, args.models_dir, promote=promote)
    print(f"artifact: {path}")

    if promote and args.api_url:
        reload_api(args.api_url, args.admin_token)

    return 0 if promote else 2


if __name__ == "__main__":
    sys.exit(main())
