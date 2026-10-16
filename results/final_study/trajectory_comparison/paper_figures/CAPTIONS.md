# Article figure pack

**Recommended main figure: Fig01_trajectory_comparison.pdf.** Standalone panels/alternatives follow as Fig02–Fig07.

All figures are 178 mm wide, with editable text in SVG and embedded TrueType fonts in PDF. PNG and LZW-compressed TIFF are exported at 600 dpi. A seven-page PDF figure book is included.

These are numerical PINN/FEM comparisons, not experimental validation. The saved numerical trajectories and errors were not changed. The input hashes are in manifest.json.

## Captions

### Fig01_trajectory_comparison: Trajectory comparison
Baseline particle comparison between PINN + ForceNet and the independent FEM + Stokes reference. (a,b) Axial and transverse positions paired at the same physical times within each path pair’s common interval. Dashed lines indicate identity; gray bands are ±5% of channel width W, not confidence intervals. (c) Synchronous-position and symmetric geometric RMS errors, normalized by W. Hatched bars indicate different endpoint classes. (d) The η₀ = 0.65 case retains the complete reference path after the predicted sticking event. Points are temporally correlated samples from seven trajectories, not independent experimental observations.

### Fig02_position_parity: Position parity
PINN/FEM position parity for seven baseline inlet seeds (1,001 matched times per path). Color denotes inlet position. Coordinate RMSE is pooled over these sampled coordinate pairs and differs from the time-integrated vector RMS. All plotted samples and common-time limits are retained; no time stretching or post-termination extrapolation is used. Identity lines and ±5% W reference corridors are shown.

### Fig03_error_bars: Trajectory error
Baseline RMS errors in synchronous position and full-path geometry. All values are percentages of channel width. The two metrics answer different questions; the geometric metric ignores time ordering. Hatching identifies the sticking/exit disagreement. The dotted 5% W line is a scale/reference line, not a universal engineering tolerance.

### Fig04_timing_and_coverage: Timing and coverage
Baseline termination-time parity and common-time coverage. The red cross and hatched bar indicate different terminal events. Termination times for different event classes are displayed but must not be interpreted as equivalent outlet residence times. The 95% coverage line provides a visual reference.

### Fig05_paired_paths: Paired particle paths
(a) Full baseline paths, with matched seed colors and solid PINN / dashed FEM lines. (b) Enlarged endpoint interpretation for the η₀ = 0.65 case. The red marker identifies predicted sticking; the hollow marker identifies reference exit. Complete stored paths are shown, including the reference portion outside the common-time interval.

### Fig06_case_heatmaps: Case comparison
Synchronous and geometric RMS errors for 42 unique parameter/seed pairs. The density-ratio-1000 control duplicates the baseline and is omitted once (seven repeated pairs). Each cell is one deterministic inlet seed; stars and red borders identify endpoint-class disagreements. Both panels share the same linear color scale. Diameter and density ratio are nondimensional.

### Fig07_error_history: Position error history
Synchronous Euclidean position error divided by width W for each baseline trajectory pair. Every curve stops at the end of its common-time interval. The red cross marks the pair with different endpoint classes. The dotted 5% W line is a reference level; no prediction is extrapolated after termination.

## Data and interpretation

- `data/baseline_metrics.csv`: the seven bar/timing records, copied from the saved comparison table.
- `data/parity_points.csv`: every coordinate pair used in the parity panels.
- `data/unique_case_metrics.csv`: the 42 nonduplicate case/seed records.
- Error percentages divide by channel width W = 0.3; they are not coordinate-relative percentages.
- All 7 baseline paths are included, including the mismatched sticking/exit case.
- Synchronous RMS uses the previously computed time integral on the common interval; geometric RMS uses both complete path polylines.
- A narrow parity corridor or a small average field error alone does not certify all particle trajectories.
- Error bars are intentionally absent: inlet-seed variability is not a measured statistical confidence interval.
