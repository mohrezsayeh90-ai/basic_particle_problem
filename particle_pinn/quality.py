"""Explicit numerical screening; field, trajectory and pressure checks stay distinct."""


def assess(metrics, skip_3d=False):
    flow = metrics.get("flow", {})
    reference = metrics.get("reference", {})
    trajectory = metrics.get("trajectory", {})
    comparisons = trajectory.get("baseline_comparison", [])
    width = metrics.get("geometry", {}).get("width", 1.)
    inf = float("inf")
    difference = reference.get("penultimate_velocity_difference")
    return dict(
        analytical_fem=metrics.get("poiseuille", {}).get("fem_velocity_relative_l2", inf) < 1e-5,
        analytical_pinn_velocity=metrics.get("poiseuille", {}).get("pinn_velocity_relative_l2", inf) < .05,
        analytical_pinn_pressure=metrics.get("poiseuille", {}).get("pinn_pressure_relative_l2", inf) < .05,
        reference_solver=reference.get("solver_ok", False),
        mesh_sensitivity=difference is not None and difference < .02,
        trajectory_interpolation=reference.get("trajectory_interpolator_velocity_error", inf) < .02,
        velocity_error=flow.get("velocity_relative_l2", inf) < .05,
        pressure_error=flow.get("pressure_relative_l2", inf) < .05,
        boundary_error=flow.get("boundary_rms", inf) < .02,
        mass_flux=flow.get("flux_relative_deviation", inf) < .05,
        drag_error=metrics.get("force", {}).get("heldout_relative_l2", inf) < .05,
        trajectory_solver=trajectory.get("total", 0)>0 and trajectory.get("failures", 1)==0,
        baseline_outcomes=bool(comparisons) and all(r["outcome_agreement"] for r in comparisons),
        baseline_position=bool(comparisons) and all(r.get("rms_position_error") is not None and r["rms_position_error"]/width < .05 for r in comparisons),
        baseline_time_coverage=bool(comparisons) and all(r["common_fraction"] > .95 for r in comparisons),
        abc3d_velocity=None if skip_3d else metrics.get("benchmark3d", {}).get("velocity_relative_l2", inf) < .05,
        abc3d_pressure=None if skip_3d else metrics.get("benchmark3d", {}).get("pressure_relative_l2", inf) < .05,
        pipeline_stages=all(r["status"]=="completed" for r in metrics.get("execution_stages", []))
                        if "execution_stages" in metrics else metrics.get("quality_gate", {}).get("pipeline_stages", False))
