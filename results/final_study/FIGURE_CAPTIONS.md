# Figure captions

1. Channel geometry, interior collocation points and velocity boundary samples.
2. Flow and force network architectures; labels show hidden layer widths.
3. Flow physics/boundary loss and normalized drag training/validation loss.
4. PINN and FEM speed and mean-centered pressure with pointwise differences on a mapped plotting grid. Reference plotting values use topological interpolation; numerical validation uses direct FEM evaluation.
5. Held-out FEM/PINN velocity and pressure parity. Pressure gauges are aligned by subtracting each sample mean.
6. Tangential velocity profiles at three channel sections.
7. Section flux relative to the prescribed inlet flux.
8. Velocity mesh sensitivity relative to the finest available reference and FEM divergence norm. The zero finest-mesh difference is displayed at the plotting floor; this is not a zero error estimate.
9. Fluid-marker paths integrated from each separately labeled velocity field.
10. Baseline particle paths; crosses identify wall sticking, circles identify other endpoints. Exact endpoint outcomes are in trajectory_summary.json.
11. Particle diameter study for PINN/ForceNet and FEM/Stokes.
12. Particle density-ratio study for both approaches.
13. Fractions of exited, stuck, time-limited, backflow and failed trajectories in each scenario.
14. Planar particle paths extruded for visualization. Depth coordinates are illustrative.
15. Held-out normalized drag parity and analytical Stokes velocity relaxation.
16. Analytical plane Poiseuille profile with independent FEM and PINN profiles and velocity errors.
17. Time-resolved particle animation. Stuck particles remain fixed; exited particles leave the display.
18. Training-seed monitor loss and held-out velocity-error distributions.
19. Three-dimensional tracer paths in the forced ABC benchmark and its exact manufactured reference.
20. ABC benchmark velocity parity and physics loss. Quantitative pressure error is reported separately.
21. Central-seed baseline particle speed and hydrodynamic drag components. At wall contact, constraint forces are outside the plotted drag model.
22. Particle endpoint classification versus inlet position and diameter/width.
