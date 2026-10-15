# Particle flow study

Preset: `standard`. Geometry: `serpentine`.

## Model

Steady, incompressible Newtonian flow; dilute spherical particles; one-way Stokes drag; irreversible wall sticking at a particle-radius clearance. No particle-particle forces, lift, Brownian motion, gravity, finite-size hydrodynamics or feedback on the fluid are modeled.

The two networks have five hidden layers with 70 and 90 tanh units, respectively. Flow training uses PDE, boundary and pressure-gauge losses; FEM interior data are held out. The force network learns the specified algebraic Stokes law; it does not identify a new interaction law.

## Validation

Analytical plane Poiseuille flow checks the FEM implementation and a separate straight-channel PINN. Curved-channel reference data come from independently assembled P2/P1 Taylor-Hood finite elements. Mesh refinement, nonlinear residual, continuity, boundary errors, flux, drag and trajectory errors are exported. Reference agreement measures numerical consistency for these equations, not experimental agreement.

### poiseuille

```json
{
  "fem_velocity_relative_l2": 8.192371368289349e-10,
  "pinn_velocity_relative_l2": 0.0021705328336898335,
  "fem_pressure_relative_l2": 8.191842870464941e-10,
  "pinn_pressure_relative_l2": 0.0017966055844855723,
  "fem_solver_ok": true
}
```

### flow

```json
{
  "velocity_relative_l2": 0.015096144982268116,
  "pressure_relative_l2": 0.008625689602803936,
  "velocity_max_absolute": 0.05400762501134132,
  "training": "restored_for_independent_evaluation",
  "restored_from": "results/validated_run/models/flow_seed_42.pt",
  "checkpoint_sha256": "dc75f2a36e1aa89cdfab5fb3161a3d88ea7607193b70d96772241abd9fd4e813",
  "original_config": {
    "preset": "standard",
    "geometry": "serpentine",
    "centerline": null,
    "rin": 0.2,
    "rout": 0.5,
    "re": 1.0,
    "density_ratio": 1000.0,
    "diameter": 0.025,
    "seed": 42,
    "device": "cpu",
    "threads": 2,
    "out": "results/validated_run",
    "skip_animation": false,
    "skip_3d": false,
    "epochs": 1000,
    "force_epochs": 1000,
    "samples": 512,
    "ns": 60,
    "nn": 12,
    "seeds": 2,
    "benchmark_epochs": 800,
    "lbfgs_steps": 250,
    "refine_epochs": 1800,
    "refine_lbfgs_steps": 700
  },
  "continuity_rms": 0.1780688315629959,
  "momentum_rms": 1.8295940160751343,
  "boundary_rms": 0.004201731574358346,
  "flux_relative_deviation": 0.022656831094608254,
  "flux": [
    0.20006284833487745,
    0.20042093971227723,
    0.1972501151631042,
    0.1974037289821743,
    0.1986976050936187,
    0.20032306463153965,
    0.20146644734591243,
    0.20183276871155098,
    0.20077316388189734,
    0.1981443709623181,
    0.19546863378107834,
    0.19578380186751132,
    0.20032584935623565,
    0.2006431472584879,
    0.2003262052318229,
    0.1995946744601393,
    0.20139657509449832,
    0.2013935754912564,
    0.19978796993941064,
    0.2003175475093196,
    0.1987962287147459,
    0.19724784093934922,
    0.19891287703193752,
    0.20002483637175789,
    0.2000341818771551
  ]
}
```

### force

```json
{
  "heldout_relative_l2": 0.007172185152757193,
  "physics": "Stokes_drag",
  "training": "restored_for_independent_evaluation",
  "restored_from": "results/validated_run/models/force.pt",
  "checkpoint_sha256": "30ad82e3ac957012b7c524ebf5334000220758cd691ab97321d8c39f0bb3ac4e"
}
```

### reference

```json
{
  "ns": 90,
  "nn": 18,
  "triangles": 3026,
  "velocity_dofs": 12530,
  "pressure_dofs": 1620,
  "iterations": 13,
  "relative_update": 4.076761697282389e-09,
  "nonlinear_residual": 1.599867808494302e-10,
  "divergence_l2": 0.04128351598649171,
  "solver_ok": true,
  "penultimate_velocity_difference": 0.0017323061876575098,
  "trajectory_interpolator_velocity_error": 0.0017652159466120783
}
```

### seed_statistics

```json
{
  "velocity_relative_l2": {
    "mean": 0.014694522621820593,
    "std": 0.0005679797890971844,
    "count": 2
  },
  "pressure_relative_l2": {
    "mean": 0.008742805820053918,
    "std": 0.00016562734280875884,
    "count": 2
  }
}
```

### trajectory

```json
{
  "total": 119,
  "failures": 0,
  "max_particle_re": 0.0122053525986495,
  "baseline_comparison": [
    {
      "seed_eta": -0.65,
      "rms_position_error": 0.008003011736350627,
      "common_time": 2.5921425883938904,
      "common_fraction": 0.9886429158848834,
      "outcome_agreement": true
    },
    {
      "seed_eta": -0.43333333333333335,
      "rms_position_error": 0.006898404255853003,
      "common_time": 2.2654477649014257,
      "common_fraction": 0.9932687107064988,
      "outcome_agreement": true
    },
    {
      "seed_eta": -0.21666666666666667,
      "rms_position_error": 0.012895066775746401,
      "common_time": 2.290678204107999,
      "common_fraction": 0.9927263367234552,
      "outcome_agreement": true
    },
    {
      "seed_eta": 0.0,
      "rms_position_error": 0.023501651914230336,
      "common_time": 2.536715516865495,
      "common_fraction": 0.985933536249829,
      "outcome_agreement": true
    },
    {
      "seed_eta": 0.21666666666666667,
      "rms_position_error": 0.04825756992378432,
      "common_time": 3.041734212390562,
      "common_fraction": 0.9619037668389813,
      "outcome_agreement": true
    },
    {
      "seed_eta": 0.43333333333333346,
      "rms_position_error": 0.10491910226477183,
      "common_time": 4.014054398162223,
      "common_fraction": 0.8739175334909213,
      "outcome_agreement": true
    },
    {
      "seed_eta": 0.65,
      "rms_position_error": 0.009884288864254123,
      "common_time": 4.595905992306705,
      "common_fraction": 0.735530822788827,
      "outcome_agreement": false
    }
  ]
}
```

### benchmark3d

```json
{
  "velocity_relative_l2": 0.02427629764367054,
  "pressure_relative_l2": 0.135012197484622,
  "domain": "periodic_cube",
  "physics": "forced_ABC_flow",
  "training": "PDE_and_boundary_only",
  "stop_reason": "training_budget",
  "lbfgs_evaluations": 0,
  "geometry_scope": "separate_3D_benchmark_not_finite_depth_channel",
  "restored_from": "results/validated_run/models/abc3d.pt",
  "checkpoint_sha256": "950ca81e05bbdf95b0c97f37b6c2dbc0759942a7d2e8e557787723895738e4b7"
}
```

## Accuracy checks

Thresholds are numerical screening choices, not evidence of experimental validity.

| Check | Result |
|---|---|
| analytical fem | True |
| analytical pinn velocity | True |
| analytical pinn pressure | True |
| reference solver | True |
| mesh sensitivity | True |
| trajectory interpolation | True |
| velocity error | True |
| pressure error | True |
| boundary error | True |
| mass flux | True |
| drag error | True |
| trajectory solver | True |
| baseline outcomes | False |
| baseline position | False |
| baseline time coverage | False |
| abc3d velocity | True |
| abc3d pressure | False |
| pipeline stages | True |

The run always exports available finite predictions, even when these checks fail. PINN and FEM panels remain separately labeled. An unsuccessful numerical trajectory is recorded as failed, never replaced with a reference trajectory. Use `metrics.json`, not a figure title, to assess accuracy.

## Scope

Every channel/Re case is trained separately. Parameter input slots alone do not establish generalization to unseen Reynolds numbers or geometries. Custom centerlines must be smooth, planar, constant-width and nonintersecting; arbitrary topologies are not supported. The 3D channel image is an extrusion for display. The separate forced ABC benchmark solves genuinely three-dimensional equations against an exact manufactured solution; it does not validate a finite-depth 3D channel.

Stokes drag assumes particle Reynolds number much smaller than one and particles small compared with channel width. Review `max_particle_re` and diameter/width for each path. One-way coupling requires dilute loading, which must be checked for a physical experiment.

All variables are nondimensional: x*=x/L₀, u*=u/U₀, t*=tU₀/L₀, p*=p L₀/(μU₀), Re=ρf U₀L₀/μ, and F*=F/[ρf(πdₚ³/6)U₀²/L₀]. Rin/Rout define nominal arc radii; for custom paths they set width Rout−Rin. Physical units require choosing L₀, U₀, ρf and μ consistently.

## Files

`figures/`: PNG and vector PDF; architecture SVG; optional GIF. `data/`: field, loss, mesh, force and trajectory CSVs. `models/`: network state dictionaries. `config.json`, `metrics.json`, `manifest.json`: settings, outcomes and provenance.

## Execution

Completed stages: 35. Failed stages: 0. Trajectories: 119.


## References

- Raissi, Perdikaris & Karniadakis (2019), Physics-informed neural networks, J. Comput. Phys. 378, 686–707. doi:10.1016/j.jcp.2018.10.045.
- Taylor & Hood (1973), A numerical solution of the Navier–Stokes equations using the finite element technique, Computers & Fluids 1, 73–100. doi:10.1016/0045-7930(73)90027-3.
- Maxey & Riley (1983), Equation of motion for a small rigid sphere in a nonuniform flow, Physics of Fluids 26, 883–889. doi:10.1063/1.864230. Only its heavy-particle Stokes-drag simplification is used here.
- Roache (1998), Verification and Validation in Computational Science and Engineering. Mesh differences here are sensitivity estimates, not a claimed GCI or formal uncertainty bound.
