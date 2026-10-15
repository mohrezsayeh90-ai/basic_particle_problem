#!/usr/bin/env python3
"""Run the complete, reproducible PINN / FEM / particle / figure pipeline."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
from scipy.integrate import solve_ivp

from particle_pinn.geometry import Channel
from particle_pinn.fem import solve_flow, exact_poiseuille, relative_error
from particle_pinn.models import FlowNet, ForceNet, train_flow, train_force, flow_diagnostics, tensor
from particle_pinn.trajectories import integrate, track_error, stokes_acceleration
from particle_pinn import figures
from particle_pinn.benchmark3d import run_benchmark, plot_benchmark
from particle_pinn.quality import assess


PRESETS = {
    "quick": dict(epochs=160, force_epochs=250, samples=192, ns=44, nn=10, seeds=2, benchmark_epochs=100, lbfgs_steps=40, refine_epochs=0, refine_lbfgs_steps=0),
    "standard": dict(epochs=1800, force_epochs=1500, samples=768, ns=84, nn=16, seeds=3, benchmark_epochs=1200, lbfgs_steps=220, refine_epochs=1000, refine_lbfgs_steps=500),
    "paper": dict(epochs=7000, force_epochs=4000, samples=1536, ns=144, nn=24, seeds=5, benchmark_epochs=5000, lbfgs_steps=600, refine_epochs=2000, refine_lbfgs_steps=1000),
}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, obj):
    path.write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean_json(rows))


def read_history(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [{k: (v if k=="optimizer" else int(v) if k=="epoch" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


class Pipeline:
    def __init__(self, out):
        self.out = out
        self.failures = []
        self.stages = []

    def stage(self, name, action):
        print(f"\n[{name}]", flush=True)
        start = time.time()
        try:
            result = action()
            self.stages.append(dict(stage=name, status="completed", seconds=time.time()-start))
            return result
        except Exception as error:
            # Keep independent stages running; the report and exit status preserve
            # the failure. No replacement numerical data are synthesized.
            self.failures.append(dict(stage=name, error=str(error), traceback=traceback.format_exc()))
            self.stages.append(dict(stage=name, status="failed", seconds=time.time()-start))
            print(f"Stage failed: {name}: {error}", file=sys.stderr, flush=True)
            write_json(self.out/"failures.json", self.failures)
            return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="standard")
    parser.add_argument("--geometry", choices=["straight", "bend", "serpentine"], default="serpentine")
    parser.add_argument("--centerline", type=Path, help="Custom x,y CSV, header required")
    parser.add_argument("--rin", type=float, default=.2)
    parser.add_argument("--rout", type=float, default=.5)
    parser.add_argument("--re", type=float, default=1.)
    parser.add_argument("--density-ratio", type=float, default=1000.)
    parser.add_argument("--diameter", type=float, default=.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--weights-from", type=Path, help="Re-evaluate a completed compatible run without retraining")
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--skip-3d", action="store_true", help="Skip the separate 3D manufactured benchmark")
    for key in ("epochs", "force_epochs", "samples", "ns", "nn", "seeds", "benchmark_epochs", "lbfgs_steps", "refine_epochs", "refine_lbfgs_steps"):
        parser.add_argument("--"+key.replace("_", "-"), type=int)
    args = parser.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if min(args.epochs, args.force_epochs, args.samples, args.seeds, args.threads, args.benchmark_epochs) < 1:
        parser.error("Training counts, samples, seeds and threads must be positive.")
    if args.ns < 12 or args.nn < 5:
        parser.error("Require ns >= 12 and nn >= 5 for mesh refinement.")
    if min(args.lbfgs_steps, args.refine_epochs, args.refine_lbfgs_steps) < 0:
        parser.error("Refinement and L-BFGS counts must be nonnegative.")
    if args.re <= 0 or args.density_ratio <= 0 or args.diameter <= 0:
        parser.error("Re, density ratio, and diameter must be positive.")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu.")
    if args.out is None:
        args.out = Path("results") / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty. Choose a new --out to preserve prior runs.")
    if args.weights_from:
        if not (args.weights_from/"metrics.json").exists():
            parser.error("--weights-from must name a completed run with metrics.json.")
        previous = json.loads((args.weights_from/"config.json").read_text())
        for key in ("geometry", "rin", "rout", "re", "centerline", "seed", "seeds"):
            if str(previous.get(key)) != str(getattr(args, key)):
                parser.error(f"Checkpoint compatibility mismatch for {key}; use its original case settings.")
    return args


def field_errors(pred, ref):
    return dict(velocity_relative_l2=relative_error(pred[:, :2], ref[:, :2]),
                pressure_relative_l2=relative_error(pred[:, 2]-pred[:, 2].mean(), ref[:, 2]-ref[:, 2].mean()),
                velocity_max_absolute=float(np.max(np.linalg.norm(pred[:, :2]-ref[:, :2], axis=1))))


def report(out, config, metrics, stages, failures, track_count):
    gate = metrics.get("quality_gate", {})
    lines = ["# Particle flow study", "", f"Preset: `{config['preset']}`. Geometry: `{config['geometry']}`.",
             "", "## Model", "", "Steady, incompressible Newtonian flow; dilute spherical particles; one-way Stokes drag; "
             "irreversible wall sticking at a particle-radius clearance. No particle-particle forces, lift, "
             "Brownian motion, gravity, finite-size hydrodynamics or feedback on the fluid are modeled.",
             "", "The two networks have five hidden layers with 70 and 90 tanh units, respectively. "
             "Flow training uses PDE, boundary and pressure-gauge losses; FEM interior data are held out. "
             "The force network learns the specified algebraic Stokes law; it does not identify a new interaction law.",
             "", "## Validation", "", "Analytical plane Poiseuille flow checks the FEM implementation and a separate straight-channel PINN. "
             "Curved-channel reference data come from independently assembled P2/P1 Taylor-Hood finite elements. "
             "Mesh refinement, nonlinear residual, continuity, boundary errors, flux, drag and trajectory errors are exported. "
             "Reference agreement measures numerical consistency for these equations, not experimental agreement.", ""]
    for name in ("poiseuille", "flow", "force", "reference", "seed_statistics", "trajectory", "benchmark3d"):
        if name in metrics:
            lines += [f"### {name}", "", "```json", json.dumps(clean_json(metrics[name]), indent=2), "```", ""]
    lines += ["## Accuracy checks", "", "Thresholds are numerical screening choices, not evidence of experimental validity.", "",
              "| Check | Result |", "|---|---|"]
    lines += [f"| {k.replace('_', ' ')} | {v} |" for k, v in gate.items()]
    lines += ["", "The run always exports available finite predictions, even when these checks fail. "
              "PINN and FEM panels remain separately labeled. An unsuccessful numerical trajectory is recorded as failed, "
              "never replaced with a reference trajectory. Use `metrics.json`, not a figure title, to assess accuracy.",
              "", "## Scope", "", "Every channel/Re case is trained separately. Parameter input slots alone do not establish "
              "generalization to unseen Reynolds numbers or geometries. Custom centerlines must be smooth, planar, "
              "constant-width and nonintersecting; arbitrary topologies are not supported. The 3D channel image is an "
              "extrusion for display. The separate forced ABC benchmark solves genuinely three-dimensional equations "
              "against an exact manufactured solution; it does not validate a finite-depth 3D channel.",
              "", "Stokes drag assumes particle Reynolds number much smaller than one and particles small compared "
              "with channel width. Review `max_particle_re` and diameter/width for each path. One-way coupling "
              "requires dilute loading, which must be checked for a physical experiment.",
              "", "All variables are nondimensional: x*=x/L₀, u*=u/U₀, t*=tU₀/L₀, "
              "p*=p L₀/(μU₀), Re=ρf U₀L₀/μ, and F*=F/[ρf(πdₚ³/6)U₀²/L₀]. "
              "Rin/Rout define nominal arc radii; for custom paths they set width Rout−Rin. "
              "Physical units require choosing L₀, U₀, ρf and μ consistently.",
              "", "## Files", "", "`figures/`: PNG and vector PDF; architecture SVG; optional GIF. "
              "`data/`: field, loss, mesh, force and trajectory CSVs. `models/`: network state dictionaries. "
              "`config.json`, `metrics.json`, `manifest.json`: settings, outcomes and provenance.",
              "", "## Execution", "", f"Completed stages: {sum(s['status']=='completed' for s in stages)}. "
              f"Failed stages: {len(failures)}. Trajectories: {track_count}.", ""]
    if failures:
        lines += ["Failures and tracebacks are recorded in `failures.json`.", ""]
        lines += [f"- {f['stage']}: {f['error']}" for f in failures]
    lines += ["", "## References", "",
              "- Raissi, Perdikaris & Karniadakis (2019), Physics-informed neural networks, J. Comput. Phys. 378, 686–707. doi:10.1016/j.jcp.2018.10.045.",
              "- Taylor & Hood (1973), A numerical solution of the Navier–Stokes equations using the finite element technique, Computers & Fluids 1, 73–100. doi:10.1016/0045-7930(73)90027-3.",
              "- Maxey & Riley (1983), Equation of motion for a small rigid sphere in a nonuniform flow, Physics of Fluids 26, 883–889. doi:10.1063/1.864230. Only its heavy-particle Stokes-drag simplification is used here.",
              "- Roache (1998), Verification and Validation in Computational Science and Engineering. Mesh differences here are sensitivity estimates, not a claimed GCI or formal uncertainty bound."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("figures", "data", "models"):
        (args.out/name).mkdir(exist_ok=True)
    out, data, figs = args.out, args.out/"data", args.out/"figures"
    config = vars(args).copy()
    write_json(out/"config.json", config)
    pipe = Pipeline(out)
    metrics, tracks = {}, []
    started = time.time()
    root = Path(__file__).resolve().parent
    sources = sorted((root/"particle_pinn").glob("*.py"))+[Path(__file__).resolve()]
    source_hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}

    def fit_channel(geometry, seed, role="flow"):
        if args.weights_from:
            name = "poiseuille.pt" if role=="poiseuille" else f"flow_seed_{seed}.pt"
            loss_name = "poiseuille_loss.csv" if role=="poiseuille" else f"flow_loss_seed_{seed}.csv"
            checkpoint = args.weights_from/"models"/name
            model = FlowNet(geometry, args.re).to(args.device)
            model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
            metadata = dict(training="restored_for_independent_evaluation", restored_from=str(checkpoint),
                            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            original_config=json.loads((args.weights_from/"config.json").read_text()))
            return model.eval(), read_history(args.weights_from/"data"/loss_name), metadata
        model, history, diag = train_flow(geometry, args.re, args.epochs, args.samples, seed,
                                          args.device, lbfgs_steps=args.lbfgs_steps)
        if args.refine_epochs:
            refined, more, second = train_flow(geometry, args.re, args.refine_epochs, max(1024, args.samples),
                                               seed, args.device, initial=model.state_dict(),
                                               lbfgs_steps=args.refine_lbfgs_steps)
            offset = max(row["epoch"] for row in history)
            for row in more:
                row["epoch"] += offset
            history += more
            diag = dict(**second, initial_phase=diag)
            model = refined
        return model, history, diag
    channel = pipe.stage("geometry", lambda: Channel(args.geometry, args.rin, args.rout, str(args.centerline) if args.centerline else None))
    pipe.stage("architecture figure", lambda: figures.architecture(figs))
    if channel is not None:
        if args.centerline:
            (data/"centerline.csv").write_bytes(args.centerline.read_bytes())
        metrics["geometry"] = dict(kind=channel.kind, length=channel.length, width=2*channel.half_width,
                                   rin=args.rin, rout=args.rout)
        pipe.stage("geometry figure", lambda: figures.geometry(figs, channel, args.seed))

    # Independent analytical verification, even if the selected geometry is invalid.
    def validate_poiseuille():
        straight = Channel("straight", args.rin, args.rout)
        fem = solve_flow(straight, args.re, ns=28, nn=8)
        model, hist, diag = fit_channel(straight, args.seed+100, "poiseuille")
        eta = np.linspace(-.98, .98, 101)
        xy = straight.map(np.ones(len(eta)), eta)
        exact, ref, pred = exact_poiseuille(xy, straight.half_width), fem.evaluate(xy), model.evaluate(xy)
        # A single cross-section has constant analytical pressure. Relative
        # pressure errors must use points spanning x, never a zero-variance slice.
        pxy = straight.sample(600, np.random.default_rng(args.seed+201), margin=.05)
        pe, pf, pp = exact_poiseuille(pxy, straight.half_width)[:, 2], fem.evaluate(pxy)[:, 2], model.evaluate(pxy)[:, 2]
        result = dict(fem_velocity_relative_l2=relative_error(ref[:, :2], exact[:, :2]),
                      pinn_velocity_relative_l2=relative_error(pred[:, :2], exact[:, :2]),
                      fem_pressure_relative_l2=relative_error(pf-pf.mean(), pe-pe.mean()),
                      pinn_pressure_relative_l2=relative_error(pp-pp.mean(), pe-pe.mean()),
                      fem_solver_ok=fem.diagnostics["solver_ok"])
        figures.poiseuille(figs, eta, exact[:, 0], ref[:, 0], pred[:, 0])
        write_csv(data/"poiseuille.csv", [dict(eta=e, exact_u=a, fem_u=b, pinn_u=c) for e, a, b, c in zip(eta, exact[:, 0], ref[:, 0], pred[:, 0])])
        write_csv(data/"poiseuille_pressure.csv", [dict(x=pt[0], y=pt[1], exact_p=a, fem_p=b, pinn_p=c) for pt, a, b, c in zip(pxy, pe, pf, pp)])
        write_csv(data/"poiseuille_loss.csv", hist)
        torch.save(model.state_dict(), out/"models"/"poiseuille.pt")
        return result
    result = pipe.stage("Poiseuille verification", validate_poiseuille)
    if result is not None:
        metrics["poiseuille"] = result

    if channel is not None:
        xy = channel.sample(1600, np.random.default_rng(args.seed+10), margin=.08)
        reference, femfield, ensemble = None, None, []
        mesh_rows, solutions = [], []
        levels = [(max(12, args.ns//2), max(5, args.nn//2)), (args.ns, args.nn),
                  (int(args.ns*1.5), int(args.nn*1.5))]
        for ns, nn in levels:
            sol = pipe.stage(f"FEM mesh {ns} x {nn}", lambda ns=ns, nn=nn: solve_flow(channel, args.re, ns, nn))
            if sol is not None:
                solutions.append(sol)
        if solutions:
            reference = solutions[-1]
            ref = pipe.stage("reference sampling", lambda: reference.evaluate(xy))
            femfield = pipe.stage("reference interpolator", reference.fast_field)
            if ref is not None:
                for sol in solutions:
                    values = pipe.stage(f"mesh comparison {sol.diagnostics['ns']}", lambda sol=sol: sol.evaluate(xy))
                    if values is not None:
                        row = {k: v for k, v in sol.diagnostics.items() if k != "updates"}
                        row["velocity_error_to_finest"] = relative_error(values[:, :2], ref[:, :2])
                        mesh_rows.append(row)
                write_csv(data/"mesh_sensitivity.csv", mesh_rows)
                pipe.stage("mesh figures", lambda: figures.refinement(figs, mesh_rows))
                metrics["reference"] = {k: v for k, v in reference.diagnostics.items() if k != "updates"}
                metrics["reference"]["penultimate_velocity_difference"] = mesh_rows[-2]["velocity_error_to_finest"] if len(mesh_rows)>=3 else None
                # Measure error introduced by the fast trajectory interpolator.
                if femfield is not None:
                    fast = femfield(xy)
                    metrics["reference"]["trajectory_interpolator_velocity_error"] = relative_error(fast[:, :2], ref[:, :2])
                np.savez_compressed(data/"fem_reference.npz", points=reference.ub.mesh.p,
                                    triangles=reference.ub.mesh.t, velocity=reference.velocity,
                                    pressure=reference.pressure, sample_xy=xy, sample_values=ref)

        flow_model, flow_hist = None, []
        for i in range(args.seeds):
            seed = args.seed+i
            result = pipe.stage(f"flow PINN seed {seed}", lambda seed=seed: fit_channel(channel, seed))
            if result is None:
                continue
            model, hist, diag = result
            torch.save(model.state_dict(), out/"models"/f"flow_seed_{seed}.pt")
            write_csv(data/f"flow_loss_seed_{seed}.csv", hist)
            pred = model.evaluate(xy)
            if reference is not None and ref is not None:
                err = field_errors(pred, ref)
                ensemble.append((seed, xy, pred, ref, hist))
            else:
                err = {}
            metrics.setdefault("seeds", []).append(dict(seed=seed, **err, **diag))
            # Prespecified first successful seed, never cherry-pick by reference error.
            if flow_model is None:
                flow_model, flow_hist = model, hist
                metrics["flow"] = dict(**err, **diag)
                diagnostics = pipe.stage("PINN physical diagnostics", lambda: flow_diagnostics(model, channel, args.re, args.seed+999))
                if diagnostics is not None:
                    metrics["flow"].update(diagnostics)

        def fit_force():
            if args.weights_from:
                checkpoint = args.weights_from/"models"/"force.pt"
                force = ForceNet(channel, 2*channel.length/channel.half_width**2).to(args.device)
                force.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
                rng = np.random.default_rng(args.seed+190)
                flow, vp = rng.uniform(-1.3, 1.3, (2048, 3)), rng.uniform(-1.3, 1.3, (2048, 2))
                flow[:, 2] *= 2*channel.length/channel.half_width**2
                inputs = np.column_stack((flow, rng.uniform(.04, .4, (2048, 1))*channel.half_width,
                                          channel.sample(2048, rng), vp))
                with torch.no_grad():
                    pred = force.normalized_drag(tensor(inputs, args.device)).cpu().numpy()
                diag = dict(heldout_relative_l2=relative_error(pred, flow[:, :2]-vp), physics="Stokes_drag",
                            training="restored_for_independent_evaluation", restored_from=str(checkpoint),
                            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                return force.eval(), read_history(args.weights_from/"data"/"force_loss.csv"), diag
            return train_force(channel, args.force_epochs, args.seed, args.device)
        force_result = pipe.stage("force PINN", fit_force)
        force_model, force_hist = None, []
        if force_result is not None:
            force_model, force_hist, metrics["force"] = force_result
            torch.save(force_model.state_dict(), out/"models"/"force.pt")
            write_csv(data/"force_loss.csv", force_hist)

        if flow_model is not None:
            field = flow_model.evaluate
            pred = field(xy)
            if reference is not None and ref is not None:
                rows = [dict(x=pt[0], y=pt[1], pinn_u=a[0], pinn_v=a[1], pinn_p=a[2],
                             fem_u=b[0], fem_v=b[1], fem_p=b[2]) for pt, a, b in zip(xy, pred, ref)]
                write_csv(data/"field_validation.csv", rows)
                def draw_fields():
                    if femfield is not None:
                        pts, ss, ee = channel.grid(140, 41)
                        a, b = field(pts), femfield(pts)
                        np.savez_compressed(data/"plotted_fields.npz", xy=pts, shape=ss.shape, predicted=a, reference=b,
                                            reference_method="topological_interpolation_of_FEM_nodal_values")
                        figures.field_panels(figs, channel, pts, a, b, shape=ss.shape)
                    else:
                        figures.field_panels(figs, channel, xy, pred, ref)
                pipe.stage("field figures", draw_fields)
                pipe.stage("parity figures", lambda: figures.parity(figs, pred, ref))
            if femfield is not None:
                pipe.stage("profile figures", lambda: figures.profiles(figs, channel, field, femfield))
                pipe.stage("flux figures", lambda: figures.flux(figs, channel, field, femfield))
            if ensemble:
                pipe.stage("seed figures", lambda: figures.ensemble(figs, channel, ensemble))
                seed_rows = [dict(seed=row["seed"], velocity_relative_l2=row["velocity_relative_l2"],
                                  pressure_relative_l2=row["pressure_relative_l2"]) for row in metrics["seeds"]
                             if "velocity_relative_l2" in row]
                write_csv(data/"seed_summary.csv", seed_rows)
                metrics["seed_statistics"] = {key: dict(mean=float(np.mean([r[key] for r in seed_rows])),
                                                       std=float(np.std([r[key] for r in seed_rows], ddof=1)) if len(seed_rows)>1 else None,
                                                       count=len(seed_rows))
                                              for key in ("velocity_relative_l2", "pressure_relative_l2")}
            if force_model is not None:
                pipe.stage("loss figures", lambda: figures.losses(figs, flow_hist, force_hist))

        # Analytical particle-relaxation check is independent of any learned field.
        def validate_drag():
            diameter, rho = args.diameter, args.density_ratio
            tau = rho*args.re*diameter**2/18
            times = np.linspace(0, 6*tau, 150)
            sol = solve_ivp(lambda t, y: stokes_acceleration([1., 0.], y, diameter, args.re, rho),
                            (0, times[-1]), [0., 0.], t_eval=times, method="Radau", rtol=1e-9, atol=1e-11)
            analytical = 1-np.exp(-times/tau)
            metrics["relaxation"] = dict(max_absolute_error=float(np.max(abs(sol.y[0]-analytical))),
                                          tau=tau, solver_ok=sol.success)
            rng = np.random.default_rng(args.seed+77)
            flow = rng.uniform(-1, 1, (1000, 3))
            flow[:, 2] *= 2*channel.length/channel.half_width**2
            diameters = rng.uniform(.04, .4, (1000, 1))*channel.half_width
            points, vp = channel.sample(1000, rng), rng.uniform(-1, 1, (1000, 2))
            inputs = np.column_stack((flow, diameters, points, vp))
            exact = flow[:, :2]-vp
            if force_model is not None:
                with torch.no_grad():
                    pred = force_model.normalized_drag(tensor(inputs, args.device)).cpu().numpy()
                figures.drag_validation(figs, pred, exact, times, sol.y[0], analytical)
                write_csv(data/"drag_validation.csv", [dict(exact_x=a[0], exact_y=a[1], predicted_x=b[0], predicted_y=b[1]) for a, b in zip(exact, pred)])
            write_csv(data/"relaxation.csv", [dict(t=t, numerical=a, analytical=b) for t, a, b in zip(times, sol.y[0], analytical)])
        pipe.stage("drag verification", validate_drag)

        def particle_study():
            available = []
            if flow_model is not None and force_model is not None:
                available.append(("PINN + ForceNet", flow_model.evaluate, force_model))
            if femfield is not None:
                available.append(("FEM + Stokes", femfield, None))
            seed_positions = np.linspace(-.65, .65, 3 if args.preset == "quick" else 7)
            baseline_d = args.diameter
            # Scale study diameters with channel width to keep admissible seeds.
            diameters = np.array([.07, .17, .30])*channel.half_width
            densities = np.array([.2, 1., 3.])*args.density_ratio
            cases = [("baseline", baseline_d, args.density_ratio)]
            cases += [(f"diameter_{d:.5g}", float(d), args.density_ratio) for d in diameters]
            cases += [(f"density_{d:.5g}", baseline_d, float(d)) for d in densities]
            for name, selected_field, force in available:
                for case, diameter, density in cases:
                    for eta in seed_positions:
                        track = integrate(channel, selected_field, args.re, diameter, density, float(eta),
                                          force, source=name, case=case)
                        tracks.append(track)
                    print(f"Paths: {name}, {case}", flush=True)
            # Flow-only integration isolates force-surrogate errors on the baseline.
            if flow_model is not None:
                for eta in seed_positions:
                    tracks.append(integrate(channel, flow_model.evaluate, args.re, baseline_d, args.density_ratio,
                                            float(eta), source="PINN + Stokes", case="baseline"))
            for source, selected_field, _ in available:
                for eta in seed_positions:
                    tracks.append(integrate(channel, selected_field, args.re, baseline_d, args.density_ratio,
                                            float(eta), tracer=True, source=source.split(" +")[0]+" tracer", case="tracer"))
        pipe.stage("particle studies", particle_study)
        if tracks:
            rows = []
            for i, t in enumerate(tracks):
                for time_value, state in zip(t.t, t.state):
                    rows.append(dict(track=i, source=t.source, case=t.case, eta=t.seed_eta,
                                     diameter=t.diameter, density_ratio=t.density_ratio, outcome=t.outcome,
                                     t=time_value, x=state[0], y=state[1], up=state[2], vp=state[3]))
            write_csv(data/"trajectories.csv", rows)
            write_json(data/"trajectory_summary.json", [{k: v for k, v in asdict(t).items() if k not in ("t", "state")} for t in tracks])
            def export_dynamics():
                dynamics, central = [], []
                for i, t in enumerate(tracks):
                    if t.case == "tracer":
                        continue
                    selected_field = flow_model.evaluate if t.source.startswith("PINN") else femfield
                    if selected_field is None:
                        continue
                    fluid = selected_field(t.state[:, :2])
                    if t.source == "PINN + ForceNet":
                        inputs = np.column_stack((fluid, np.full(len(t.t), t.diameter), t.state[:, :2], t.state[:, 2:]))
                        with torch.no_grad():
                            forces = force_model.force(tensor(inputs, args.device), args.re).cpu().numpy()
                    else:
                        forces = 18*(fluid[:, :2]-t.state[:, 2:])/(args.re*t.diameter**2)
                    for tt, state, uvp, force in zip(t.t, t.state, fluid, forces):
                        row = dict(track=i, source=t.source, case=t.case, eta=t.seed_eta, t=tt,
                                   x=state[0], y=state[1], up=state[2], vp=state[3], uf=uvp[0], vf=uvp[1],
                                   Fx=force[0], Fy=force[1], particle_re=args.re*t.diameter*np.linalg.norm(uvp[:2]-state[2:]))
                        dynamics.append(row)
                        if t.case=="baseline" and abs(t.seed_eta)<1e-8:
                            central.append(row)
                write_csv(data/"particle_dynamics.csv", dynamics)
                if central:
                    figures.particle_dynamics(figs, central)
                figures.deposition(figs, channel, tracks)
            pipe.stage("particle dynamics exports", export_dynamics)
            baseline = [t for t in tracks if t.case == "baseline" and t.source != "PINN + Stokes"]
            pipe.stage("baseline paths figure", lambda: figures.track_panels(figs, channel, baseline, "10_particle_paths", "Particle paths"))
            pipe.stage("tracer figure", lambda: figures.track_panels(figs, channel, [t for t in tracks if t.case=="tracer"], "09_fluid_paths", "Fluid paths"))
            ds = [t for t in tracks if t.case.startswith("diameter")]
            rs = [t for t in tracks if t.case.startswith("density")]
            pipe.stage("diameter figures", lambda: figures.parameter_study(figs, channel, ds, "diameter", "11_diameter_study"))
            pipe.stage("density figures", lambda: figures.parameter_study(figs, channel, rs, "density_ratio", "12_density_study"))
            pipe.stage("outcome figure", lambda: figures.outcomes(figs, [t for t in tracks if t.case != "tracer"]))
            pipe.stage("extrusion figure", lambda: figures.three_dimensional(figs, channel, [t for t in baseline if t.source=="FEM + Stokes"]))
            if not args.skip_animation:
                pipe.stage("particle animation", lambda: figures.motion_animation(figs, channel, baseline))
            comparisons = []
            for t in baseline:
                if t.source != "PINN + ForceNet":
                    continue
                other = next((r for r in tracks if r.source=="FEM + Stokes" and r.case==t.case and r.seed_eta==t.seed_eta), None)
                isolated = next((r for r in tracks if r.source=="PINN + Stokes" and r.case==t.case and r.seed_eta==t.seed_eta), None)
                if other is not None:
                    comparisons.append(dict(seed_eta=t.seed_eta, **track_error(t, other)))
                if isolated is not None:
                    metrics.setdefault("force_trajectory_isolation", []).append(dict(seed_eta=t.seed_eta, **track_error(t, isolated)))
            metrics["trajectory"] = dict(total=len(tracks), failures=sum(t.outcome=="integration_failed" for t in tracks),
                                          max_particle_re=max((t.max_particle_re for t in tracks if np.isfinite(t.max_particle_re)), default=None),
                                          baseline_comparison=comparisons)

    if not args.skip_3d:
        def benchmark():
            if args.weights_from:
                checkpoint = args.weights_from/"models"/"abc3d.pt"
                result = list(run_benchmark(0, args.seed, args.device, lbfgs_steps=0,
                                           initial=torch.load(checkpoint, map_location=args.device, weights_only=True)))
                result[-1] = read_history(args.weights_from/"data"/"benchmark3d_loss.csv")
                result[-2]["restored_from"] = str(checkpoint)
                result[-2]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                return tuple(result)
            return run_benchmark(args.benchmark_epochs, args.seed, args.device,
                                 lbfgs_steps=30 if args.preset=="quick" else 400 if args.preset=="standard" else 1000)
        result3d = pipe.stage("3D ABC benchmark", benchmark)
        if result3d is not None:
            model, xyz, pred, exact, paths, metrics["benchmark3d"], hist = result3d
            torch.save(model.state_dict(), out/"models"/"abc3d.pt")
            np.savez_compressed(data/"benchmark3d.npz", xyz=xyz, predicted=pred, exact=exact, paths=np.array(paths))
            write_csv(data/"benchmark3d_loss.csv", hist)
            pipe.stage("3D benchmark figures", lambda: plot_benchmark(figs, result3d))

    metrics["execution_stages"] = pipe.stages
    metrics["quality_gate"] = assess(metrics, args.skip_3d)
    metrics["elapsed_seconds"] = time.time()-started
    write_json(out/"metrics.json", metrics)
    write_json(out/"stages.json", pipe.stages)
    write_json(out/"failures.json", pipe.failures)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "matplotlib", "scikit-fem", "Pillow")}
    manifest = dict(python=platform.python_version(), versions=versions,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=source_hashes,
                    artifacts=[str(f.relative_to(out)) for f in sorted(out.rglob("*")) if f.is_file()])
    write_json(out/"manifest.json", manifest)
    report(out, config, metrics, pipe.stages, pipe.failures, len(tracks))
    print(f"\nArtifacts: {out.resolve()}\nExecution failures: {len(pipe.failures)}", flush=True)
    print(json.dumps(clean_json(metrics["quality_gate"]), indent=2), flush=True)
    # Accuracy failure is a reported scientific result, not a Python execution error.
    # Numerical / implementation stage failures produce exit 2 after all exports.
    return 2 if pipe.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
