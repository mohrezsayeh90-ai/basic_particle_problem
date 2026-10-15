#!/usr/bin/env python3
"""Generate proposal tables/macros from actual repository JSON and CSV evidence.

Standard library only. No training, external data retrieval or numerical-result
modification occurs. Every required figure and numerical input is checked.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REQUIRED_FIGURES = ["01_geometry", "02_architecture", "03_losses", "04_flow_fields",
                    "05_validation_parity", "06_velocity_profiles", "07_mass_flux", "08_mesh_sensitivity",
                    "09_fluid_paths", "10_particle_paths", "11_diameter_study", "12_density_study",
                    "13_particle_outcomes", "14_extruded_paths", "15_drag_validation", "16_poiseuille_validation",
                    "18_seed_sensitivity", "19_3d_benchmark", "20_3d_validation", "21_particle_dynamics",
                    "22_deposition_map"]


def escape(value):
    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(c, c) for c in str(value))


def number(value, decimals=3):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("A required proposal metric is non-finite.")
    return f"{value:.{decimals}f}"


def scientific(value):
    mantissa, exponent = f"{float(value):.3e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def output_table(destination, rows):
    destination.write_text("% Generated from repository evidence; do not edit by hand.\n"+
                           "\n".join(" & ".join(row)+r" \\" for row in rows)+"\n", encoding="utf-8")


def generate(results, target=HERE/"generated"):
    results, target = Path(results).resolve(), Path(target).resolve()
    paths = [results/"metrics.json", results/"config.json", results/"manifest.json", results/"stages.json",
             results/"data"/"mesh_sensitivity.csv", results/"data"/"seed_summary.csv",
             results/"data"/"trajectory_summary.json"]
    paths += [results/"figures"/f"{name}.pdf" for name in REQUIRED_FIGURES]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing proposal inputs:\n"+"\n".join(missing))
    m = json.loads((results/"metrics.json").read_text())
    cfg = json.loads((results/"config.json").read_text())
    manifest = json.loads((results/"manifest.json").read_text())
    stages = json.loads((results/"stages.json").read_text())
    mesh = csv_rows(results/"data"/"mesh_sensitivity.csv")
    seeds = csv_rows(results/"data"/"seed_summary.csv")
    tracks = json.loads((results/"data"/"trajectory_summary.json").read_text())
    if len(tracks) != m["trajectory"]["total"]:
        raise ValueError("Trajectory JSON counts disagree with metrics.json.")
    if sum(t["outcome"]=="integration_failed" for t in tracks) != m["trajectory"]["failures"]:
        raise ValueError("Trajectory failure counts disagree.")
    if int(mesh[-1]["triangles"]) != m["reference"]["triangles"]:
        raise ValueError("Finest-mesh metadata disagree.")
    if len(seeds) != m["seed_statistics"]["velocity_relative_l2"]["count"]:
        raise ValueError("Seed counts disagree.")
    target.mkdir(parents=True, exist_ok=True)
    baseline = m["trajectory"]["baseline_comparison"]
    for comparison in baseline:
        paired = []
        for source in ["PINN + ForceNet", "FEM + Stokes"]:
            selected = [t for t in tracks if t["source"]==source and t["case"]=="baseline"
                        and math.isclose(t["seed_eta"],comparison["seed_eta"],abs_tol=1e-10)]
            if len(selected)!=1:
                raise ValueError("Baseline trajectory pairing is missing or ambiguous.")
            paired.append(selected[0])
        if (paired[0]["outcome"]==paired[1]["outcome"]) != comparison["outcome_agreement"]:
            raise ValueError("Baseline outcome flags disagree with the raw trajectory summary.")
    agreement = sum(t["outcome_agreement"] for t in baseline)
    width = m["geometry"]["width"]
    tracer_count = sum(t["case"]=="tracer" for t in tracks)
    macros = dict(FlowVelocityPercent=number(100*m["flow"]["velocity_relative_l2"]),
                  FlowPressurePercent=number(100*m["flow"]["pressure_relative_l2"]),
                  DragPercent=number(100*m["force"]["heldout_relative_l2"]),
                  PoiseuilleVelocityPercent=number(100*m["poiseuille"]["pinn_velocity_relative_l2"]),
                  PoiseuillePressurePercent=number(100*m["poiseuille"]["pinn_pressure_relative_l2"]),
                  ABCVelocityPercent=number(100*m["benchmark3d"]["velocity_relative_l2"]),
                  ABCPressurePercent=number(100*m["benchmark3d"]["pressure_relative_l2"]),
                  FluxPercent=number(100*m["flow"]["flux_relative_deviation"]),
                  MeshDifferencePercent=number(100*m["reference"]["penultimate_velocity_difference"]),
                  InterpolationPercent=number(100*m["reference"]["trajectory_interpolator_velocity_error"]),
                  VelocityMeanPercent=number(100*m["seed_statistics"]["velocity_relative_l2"]["mean"]),
                  VelocitySDPercent=number(100*m["seed_statistics"]["velocity_relative_l2"]["std"]),
                  MaxPathWidthPercent=number(100*max(t["rms_position_error"] for t in baseline)/width),
                  MinimumTimeCoveragePercent=number(100*min(t["common_fraction"] for t in baseline)),
                  MaximumParticleRe=number(m["trajectory"]["max_particle_re"],5),
                  BaselineAgreement=str(agreement), BaselineCount=str(len(baseline)),
                  TotalIntegrations=str(len(tracks)), ParticleIntegrations=str(len(tracks)-tracer_count),
                  TracerIntegrations=str(tracer_count), FailedIntegrations=str(m["trajectory"]["failures"]),
                  CompletedStages=str(sum(s["status"]=="completed" for s in stages)),
                  FailedStages=str(sum(s["status"]!="completed" for s in stages)),
                  ChannelWidth=number(width,2), ChannelLength=number(m["geometry"]["length"],4),
                  ReynoldsNumber=number(cfg["re"],1), InnerRadius=number(cfg["rin"],2),
                  OuterRadius=number(cfg["rout"],2), BaselineDiameter=number(cfg["diameter"],3),
                  BaselineDensity=number(cfg["density_ratio"],0),
                  FinestTriangles=str(m["reference"]["triangles"]))
    prior = target/"evidence_manifest.json"
    revision = json.loads(prior.read_text()).get("data_revision", "unversioned") if prior.exists() else "unversioned"
    try:
        rel = results.relative_to(HERE.parent)
        proc = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(rel)], cwd=HERE.parent,
                              capture_output=True, text=True, check=True)
        if proc.stdout.strip():
            revision = proc.stdout.strip()
    except (ValueError, OSError, subprocess.CalledProcessError):
        pass
    macros["EvidenceRevision"] = escape(revision[:12])
    macros["EvidenceRevisionFull"] = escape(revision)
    (target/"metrics.tex").write_text("% Generated numerical evidence.\n"+
           "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items())+"\n", encoding="utf-8")
    figure_path = os.path.relpath(results/"figures", HERE).replace(os.sep, "/")+"/"
    if any(c in figure_path for c in "{}%#\n"):
        raise ValueError("Unsupported TeX-special character in figure path.")
    (target/"paths.tex").write_text("\\graphicspath{{"+figure_path+"}}\n", encoding="utf-8")
    summary = [
        ("Curved channel", "Velocity relative error", macros["FlowVelocityPercent"]+r"\%"),
        ("Curved channel", "Pressure relative error", macros["FlowPressurePercent"]+r"\%"),
        ("Drag network", "Held-out normalized force error", macros["DragPercent"]+r"\%"),
        ("Poiseuille PINN", "Velocity relative error", macros["PoiseuilleVelocityPercent"]+r"\%"),
        ("Poiseuille PINN", "Pressure relative error", macros["PoiseuillePressurePercent"]+r"\%"),
        ("Poiseuille FEM", "Velocity relative error", scientific(m["poiseuille"]["fem_velocity_relative_l2"])),
        ("Finest FEM pair", "Velocity difference", macros["MeshDifferencePercent"]+r"\%"),
        ("Reference interpolation", "Velocity difference", macros["InterpolationPercent"]+r"\%"),
        ("Baseline particle outcomes", "Matched endpoint classes", f"{agreement}/{len(baseline)}"),
        ("ABC benchmark", "Velocity relative error", macros["ABCVelocityPercent"]+r"\%"),
        ("ABC benchmark", "Pressure relative error", macros["ABCPressurePercent"]+r"\%")]
    output_table(target/"summary.tex", summary)
    output_table(target/"mesh.tex", [(f"{r['ns']} $\\times$ {r['nn']}", r["triangles"],
                 number(100*float(r["velocity_error_to_finest"])), scientific(r["divergence_l2"])) for r in mesh])
    output_table(target/"seeds.tex", [(r["seed"], number(100*float(r["velocity_relative_l2"])),
                 number(100*float(r["pressure_relative_l2"]))) for r in seeds])
    output_table(target/"trajectories.tex", [(number(t["seed_eta"],3),number(100*t["rms_position_error"]/width),
                 number(100*t["common_fraction"],2),"Yes" if t["outcome_agreement"] else "No") for t in baseline])
    outcome_rows = []
    cases = ["baseline"]
    for prefix in ["diameter_", "density_"]:
        cases += sorted({t["case"] for t in tracks if t["case"].startswith(prefix)},key=lambda key: float(key.split("_",1)[1]))
    for case in cases:
        if case=="baseline":
            case_label = "Baseline"
        else:
            symbol = "d_p" if case.startswith("diameter_") else r"\rho_r"
            case_label = "$"+symbol+"="+case.split("_",1)[1]+"$"
        for source in ["PINN + ForceNet", "FEM + Stokes"]:
            selected = [t for t in tracks if t["source"]==source and t["case"]==case]
            counts = Counter(t["outcome"] for t in selected)
            outcome_rows.append((case_label, escape(source), str(counts["exited"]),str(counts["stuck"]),
                                 str(sum(v for k,v in counts.items() if k not in ["exited","stuck"]))))
    output_table(target/"outcomes.tex", outcome_rows)
    output_table(target/"software.tex", [(escape(k),escape(v)) for k,v in manifest["versions"].items()])
    labels = {"analytical_fem":"Analytical FEM", "analytical_pinn_velocity":"Poiseuille PINN velocity",
              "analytical_pinn_pressure":"Poiseuille PINN pressure", "reference_solver":"FEM nonlinear solve",
              "mesh_sensitivity":"Reference mesh sensitivity", "trajectory_interpolation":"Reference interpolator",
              "velocity_error":"Channel velocity", "pressure_error":"Channel pressure", "boundary_error":"Velocity boundary",
              "mass_flux":"Channel mass flux", "drag_error":"Drag surrogate", "trajectory_solver":"Trajectory integration",
              "baseline_outcomes":"Particle endpoint agreement", "baseline_position":"Particle position error",
              "baseline_time_coverage":"Trajectory time coverage", "abc3d_velocity":"ABC velocity",
              "abc3d_pressure":"ABC pressure", "pipeline_stages":"Execution stages"}
    output_table(target/"quality.tex", [(labels.get(k,escape(k)), "Met" if v is True else "Not met" if v is False else "Not evaluated")
                                       for k,v in m["quality_gate"].items()])
    definitions = []
    for name in ["summary", "mesh", "seeds", "trajectories", "outcomes", "software", "quality"]:
        rows = "\n".join(line for line in (target/f"{name}.tex").read_text().splitlines() if not line.startswith("%"))
        definitions.append("\\expandafter\\def\\csname EvidenceTable"+name+"\\endcsname{\n"+rows+"\n}")
    (target/"tables.tex").write_text("\n".join(definitions)+"\n", encoding="utf-8")
    evidence = dict(data_revision=revision, inputs={str(p.relative_to(results)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    figure_count=len(REQUIRED_FIGURES), result_directory=str(results), numerical_results_modified=False)
    (target/"evidence_manifest.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    return evidence


def default_results():
    bundled = HERE/"inputs"
    return bundled if bundled.exists() else HERE.parent/"results"/"final_study"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=default_results())
    args = parser.parse_args()
    evidence = generate(args.results)
    print(f"Prepared {evidence['figure_count']} figures and evidence tables from {args.results}.")
