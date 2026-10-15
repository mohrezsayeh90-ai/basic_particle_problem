#!/usr/bin/env python3
"""Refresh explicit accuracy checks on existing numerical results; do not alter data."""
import argparse
import hashlib
import json
from pathlib import Path
from particle_pinn.quality import assess
from run_particle_pinn import write_json, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    out = args.run
    cfg = json.loads((out/"config.json").read_text())
    metrics = json.loads((out/"metrics.json").read_text())
    stages = json.loads((out/"stages.json").read_text())
    failures = json.loads((out/"failures.json").read_text())
    metrics["execution_stages"] = stages
    metrics["quality_gate"] = assess(metrics, cfg.get("skip_3d", False))
    quality_source = Path(__file__).parent/"particle_pinn"/"quality.py"
    metrics["assessment"] = dict(method="explicit_field_trajectory_and_3D_pressure_screening",
                                source_sha256=hashlib.sha256(quality_source.read_bytes()).hexdigest(),
                                numerical_results="unchanged")
    write_json(out/"metrics.json", metrics)
    report(out, cfg, metrics, stages, failures, metrics.get("trajectory", {}).get("total", 0))
    print(json.dumps(metrics["quality_gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
