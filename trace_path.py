#!/usr/bin/env python3
"""Trace a fluid marker or particle from an arbitrary interior point in a saved run."""
import argparse
from pathlib import Path
from particle_pinn.api import load_run, trace
from particle_pinn.figures import track_panels
from run_particle_pinn import write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--start", nargs=2, required=True, type=float, metavar=("X", "Y"))
    parser.add_argument("--velocity", nargs=2, type=float, metavar=("UP", "VP"))
    parser.add_argument("--tracer", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--diameter", type=float)
    parser.add_argument("--density-ratio", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new, empty output directory.")
    run = load_run(args.run)
    track = trace(run, args.start, args.tracer, args.reference, args.diameter, args.density_ratio, args.velocity)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/"path.csv", [dict(t=t, x=y[0], y=y[1], up=y[2], vp=y[3]) for t, y in zip(track.t, track.state)])
    write_json(args.out/"outcome.json", dict(source=track.source, outcome=track.outcome,
                                            message=track.message, max_particle_re=track.max_particle_re))
    track_panels(args.out, run["channel"], [track], "path", "Fluid path" if args.tracer else "Particle path")
    print(f"{track.outcome}: {args.out.resolve()}")
    return int(track.outcome=="integration_failed")


if __name__ == "__main__":
    raise SystemExit(main())
