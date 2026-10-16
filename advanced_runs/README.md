# Parametric training snapshots

GitHub Actions saves each completed 1,000-step snapshot under
`<run-id>/snapshots/step_001000/`, `step_002000/`, and so on.
Each contains the three-case field/path figure, motion GIF, metrics,
particle coordinates and training history. `checkpoint.pt` at the run root
is the latest published full optimizer/RNG checkpoint for recovery.

The GPU trains continuously. A separate CPU process renders immutable
snapshots and publishes private Kaggle dataset versions; GitHub Actions
collects them while the same GPU session stays running. Publication can lag
the actual optimizer step. See `advanced/run_state.json` for the last
published step and active Kaggle version.
