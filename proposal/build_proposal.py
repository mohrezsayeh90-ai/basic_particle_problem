#!/usr/bin/env python3
"""Build the proposal from existing evidence and optionally bundle portable sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from generate_results import HERE, REQUIRED_FIGURES, default_results, generate


def make_bundle(destination, results, evidence):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [HERE/name for name in ["proposal.tex", "metadata.tex", "references.tex", "README.md",
                                    "generate_results.py", "build_proposal.py", ".gitignore"]]
    for directory in ["chapters", "template", "generated"]:
        files += [p for p in (HERE/directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            name = str(path.relative_to(HERE))
            if name == "generated/paths.tex":
                archive.writestr(name, "\\graphicspath{{inputs/figures/}}\n")
            elif name == "generated/evidence_manifest.json":
                snapshot = dict(evidence, result_directory="inputs")
                archive.writestr(name, json.dumps(snapshot, indent=2)+"\n")
            else:
                archive.write(path, name)
        for name in evidence["inputs"]:
            archive.write(results/name, "inputs/"+name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Source archive integrity check failed.")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=default_results())
    parser.add_argument("--bundle", action="store_true")
    parser.add_argument("--bundle-path", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    evidence = generate(results)
    if shutil.which("latexmk") is None or shutil.which("pdflatex") is None:
        raise SystemExit("Install latexmk and the TeX Live packages listed in proposal/README.md.")
    build = HERE/"build"
    build.mkdir(exist_ok=True)
    env = os.environ.copy()
    for variable, directory in [("TEXMFVAR", "texmf-var"), ("TEXMFCONFIG", "texmf-config")]:
        (build/directory).mkdir(exist_ok=True)
        env[variable] = str(build/directory)
    command = ["latexmk", "-pdf", "-pdflatex=pdflatex -no-shell-escape %O %S", "-interaction=nonstopmode",
               "-halt-on-error", "-file-line-error", "-outdir=build", "proposal.tex"]
    process = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
    transcript = process.stdout+process.stderr
    (build/"compile_output.txt").write_text(transcript, encoding="utf-8")
    if process.returncode:
        print("\n".join(transcript.splitlines()[-65:]), file=sys.stderr)
        raise SystemExit(process.returncode)
    pdf = build/"proposal.pdf"
    log = (build/"proposal.log").read_text(errors="replace")
    failures = [line for line in log.splitlines() if "Missing character:" in line or
                ("Warning:" in line and ("undefined" in line or "multiply defined" in line))]
    if failures:
        raise SystemExit("Unresolved document warnings:\n"+"\n".join(failures))
    aux = (build/"proposal.aux").read_text(errors="replace")
    labels = set(re.findall(r"\\newlabel\{(fig:[^}]+)\}", aux))
    labels = {label for label in labels if not label.endswith("@cref")}
    if len(labels) != len(REQUIRED_FIGURES):
        raise SystemExit(f"Expected {len(REQUIRED_FIGURES)} figure labels; found {len(labels)}.")
    checks = dict(compiler_exit_code=process.returncode, required_figures=len(REQUIRED_FIGURES),
                  figure_labels=len(labels), unresolved_references=failures,
                  overfull_boxes=[line for line in log.splitlines() if "Overfull \\" in line],
                  data_revision=evidence["data_revision"],
                  numerical_inputs_modified=False)
    if shutil.which("pdftotext"):
        text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        for marker in ["[TO WRITE:", "FORMATTING PLACEHOLDER", "Your Full Name", "Writing guide ---"]:
            if marker in text:
                raise SystemExit(f"Template content remains in PDF: {marker}")
        expected = json.loads((results/"metrics.json").read_text())
        for value in [expected["flow"]["velocity_relative_l2"], expected["flow"]["pressure_relative_l2"],
                      expected["benchmark3d"]["pressure_relative_l2"]]:
            if f"{100*value:.3f}" not in text:
                raise SystemExit("A required data-derived metric was not found in PDF text.")
        checks["pdf_text_checks"] = "passed"
    if shutil.which("pdfinfo"):
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        pages = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
        checks["pages"] = int(pages.group(1)) if pages else None
    destination = HERE/"Particle_PINN_Proposal.pdf"
    shutil.copy2(pdf, destination)
    checks["pdf_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (HERE/"build_validation.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    print(f"PDF: {destination}\nPages: {checks.get('pages', 'not measured')}\nFigures: {len(labels)}")
    if args.bundle or args.bundle_path:
        default_archive = HERE.parent/"downloads"/"particle_pinn_proposal_source.zip"
        if (HERE/"inputs").exists():
            default_archive = HERE/"particle_pinn_proposal_source.zip"
        archive = make_bundle(args.bundle_path or default_archive, results, evidence)
        print(f"Portable source ZIP: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
