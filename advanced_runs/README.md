# Parametric training snapshots

GitHub Actions saves each completed 1,000-step snapshot under
`<run-id>/snapshots/step_001000/`, `step_002000/`, and so on.
Each contains the three-case field/path figure, motion GIF, metrics,
particle coordinates and training history. `checkpoint.pt` at the run root
is the latest full optimizer/RNG checkpoint for the next Kaggle chunk.

No advanced training has started merely by adding this folder.
