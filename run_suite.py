#!/usr/bin/env python3
"""Run separate channel/Re models and aggregate their actual validation metrics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_particle_pinn import write_csv, write_json
from particle_pinn import figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["quick", "standard", "paper"], default="quick")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new, empty output directory.")
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    cases = [("straight", ["--geometry", "straight"]), ("bend", ["--geometry", "bend"]),
             ("serpentine", ["--geometry", "serpentine"]),
             ("serpentine_Re3", ["--geometry", "serpentine", "--re", "3"]),
             ("custom", ["--centerline", str(root/"examples"/"centerline.csv")])]
    rows = []
    for name, options in cases:
        out = args.out/name
        cmd = [sys.executable, str(root/"run_particle_pinn.py"), "--preset", args.preset,
               "--out", str(out.resolve()), "--skip-3d", "--skip-animation", *options]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        process = subprocess.run(cmd, cwd=root, check=False)
        metrics_path = out/"metrics.json"
        m = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        flow = m.get("flow", {})
        rows.append(dict(case=name, exit_code=process.returncode,
                         velocity_error=flow.get("velocity_relative_l2"),
                         pressure_error=flow.get("pressure_relative_l2"),
                         mass_flux_error=flow.get("flux_relative_deviation")))
        write_csv(args.out/"suite_metrics.csv", rows)
        write_json(args.out/"suite_metrics.json", rows)
    fig, axes = figures.plt.subplots(1, 2, figsize=(11, 4))
    for row in rows:
        folder = args.out/row["case"]
        cfg_path = folder/"config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        from particle_pinn.geometry import Channel
        c = Channel(cfg["geometry"], cfg["rin"], cfg["rout"], cfg.get("centerline"))
        xy = c.curve(c.lookup_s)
        axes[0].plot(*xy.T, label=row["case"])
    axes[0].set(title="Channel families", xlabel="x / L₀", ylabel="y / L₀", aspect="equal")
    axes[0].legend(fontsize=8)
    valid = [r for r in rows if r["velocity_error"] is not None]
    axes[1].bar([r["case"] for r in valid], [r["velocity_error"] for r in valid])
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set(title="Case validation", ylabel="Velocity relative L₂ error")
    figures.save(fig, args.out, "suite_validation")
    return int(any(r["exit_code"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
