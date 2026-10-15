"""Load trained runs and trace from any admissible point, without retraining."""
import json
from pathlib import Path
import numpy as np
import torch
from skfem import MeshTri, Basis, ElementVector, ElementTriP2, ElementTriP1
from .geometry import Channel
from .models import FlowNet, ForceNet
from .fem import FEMResult
from .trajectories import integrate


def load_run(folder, device="cpu"):
    folder = Path(folder)
    cfg = json.loads((folder/"config.json").read_text())
    metrics = json.loads((folder/"metrics.json").read_text())
    path = cfg.get("centerline")
    retained = folder/"data"/"centerline.csv"
    if path and retained.exists():
        path = str(retained)
    channel = Channel(cfg["geometry"], cfg["rin"], cfg["rout"], path)
    models = folder/"models"
    result = dict(channel=channel, config=cfg, metrics=metrics)
    seeds = metrics.get("seeds", [])
    if seeds:
        flow = FlowNet(channel, cfg["re"]).to(device)
        flow.load_state_dict(torch.load(models/f"flow_seed_{seeds[0]['seed']}.pt", map_location=device, weights_only=True))
        result["flow"] = flow.eval()
    if (models/"force.pt").exists():
        force = ForceNet(channel, 2*channel.length/channel.half_width**2).to(device)
        force.load_state_dict(torch.load(models/"force.pt", map_location=device, weights_only=True))
        result["force"] = force.eval()
    reference_path = folder/"data"/"fem_reference.npz"
    if reference_path.exists():
        with np.load(reference_path) as ref:
            mesh = MeshTri(ref["points"], ref["triangles"])
            ub = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
            pb = Basis(mesh, ElementTriP1(), intorder=5)
            result["reference"] = FEMResult(channel, cfg["re"], ub, pb, ref["velocity"], ref["pressure"], {})
    return result


def trace(run, start, tracer=False, reference=False, diameter=None, density_ratio=None, initial_velocity=None):
    cfg, channel = run["config"], run["channel"]
    if reference:
        field, force, name = run["reference"].fast_field(), None, "FEM + Stokes"
    else:
        field, force, name = run["flow"].evaluate, run.get("force"), "PINN + ForceNet"
        if force is None and not tracer:
            raise ValueError("This run has no trained force model. Select the reference explicitly.")
    return integrate(channel, field, cfg["re"], diameter if diameter is not None else cfg["diameter"],
                     density_ratio if density_ratio is not None else cfg["density_ratio"],
                     force_model=force, tracer=tracer, source=name+ (" tracer" if tracer else ""),
                     start_xy=start, initial_velocity=initial_velocity, case="custom_start")
