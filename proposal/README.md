# Data-driven research proposal

This English Master of Science proposal adapts the supplied thesis template for
Mohammad Reza Sayah, Sharif University of Technology, with supervisor Mohammad Ali
Bijarchi. Edit `metadata.tex` to change the title-page details. Department, student
number and co-supervisor are optional and omitted when empty.

The original template is preserved byte-for-byte in `template/`. Its guide text,
sample references and completion/approval pages are not included in the proposal.
The final document retains its report class, A4 layout, margins, Latin Modern
fonts, 1.5 line spacing, blue headings, front matter and chapter structure.

## Build

From the repository root:

```bash
python proposal/build_proposal.py --bundle
```

Python 3.10+ and a TeX Live/MiKTeX installation with `pdflatex` and `latexmk` are
needed. Scientific Python packages and model retraining are not required to build
the PDF. On Ubuntu/Debian the usual packages are:

```bash
sudo apt-get install latexmk texlive-latex-base texlive-latex-recommended \
  texlive-latex-extra texlive-fonts-recommended
```

The build writes `Particle_PINN_Proposal.pdf` and `build_validation.json` here.
Temporary TeX files and the compiler transcript are under ignored `build/`.
Shell escape is disabled. Undefined references/citations, missing glyphs,
missing inputs, unexpected figure counts and missing headline metrics cause an
explicit failure. Overfull boxes are reported for review.

The `--bundle` option also creates
`../downloads/particle_pinn_proposal_source.zip`. This is a portable source archive
with all chapter files, the original template, generated tables, and the exact
input JSON/CSV and PDF figures needed for compilation. Extract it and choose
`proposal.tex` as the main document in Overleaf (pdfLaTeX). Compile repeatedly or
run `python build_proposal.py` inside the extracted directory.

## Connection to the repository results

`generate_results.py` reads, without modifying:

- `results/final_study/metrics.json`, `config.json`, `manifest.json`, `stages.json`;
- mesh and seed CSV tables and the trajectory-summary JSON;
- all 21 existing figure PDFs under `results/final_study/figures/`.

It writes numeric macros and table rows under `generated/`, records input SHA-256
hashes and the last Git revision affecting the data directory, and uses relative
figure paths. The proposal reads the repository's existing figures directly;
there are no manually retyped result tables or remote-download dependencies.
All paths in the portable archive are rewritten to its bundled `inputs/`.

For another compatible run:

```bash
python proposal/build_proposal.py --results results/NEW_RUN
```

The schema, all evaluation components and required figures must be present. The
generator checks agreement between metrics and case/mesh/seed counts. It does
not fabricate missing values or replace failed cases with reference outputs.

**Updating data does not automatically revise scientific interpretation.** The
chapter prose describes the current baseline and must be reviewed after changing
case definitions, outcomes or the scientific scope. The existing path and 3D
pressure limitations are explicitly retained. Proposed research and completed
preliminary evidence are separate chapters. The document is not a claim of
experimental validation or a completed 3D channel solution.
