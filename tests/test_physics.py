"""Numerical checks of equations, boundary events, references and network contracts."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn
from particle_pinn.geometry import Channel
from particle_pinn.fem import solve_flow, exact_poiseuille, relative_error
from particle_pinn.models import FlowNet, ForceNet, tensor, residuals
from particle_pinn.trajectories import integrate, stokes_acceleration
from particle_pinn.benchmark3d import abc_residual
from particle_pinn.quality import assess

torch.set_num_threads(1)


class GeometryTests(unittest.TestCase):
    def test_map_inverse_for_all_geometries(self):
        for kind in ("straight", "bend", "serpentine"):
            c = Channel(kind)
            s = np.linspace(.01*c.length, .99*c.length, 100)
            eta = np.linspace(-.95, .95, 100)
            a, b = c.coordinates(c.map(s, eta))
            np.testing.assert_allclose(a, s, atol=1e-5)
            np.testing.assert_allclose(b, eta, atol=1e-5)

    def test_custom_centerline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"path.csv"
            x = np.linspace(0, 3, 50)
            np.savetxt(path, np.column_stack((x, .1*np.sin(x))), delimiter=",", header="x,y", comments="")
            c = Channel(path=str(path))
            self.assertEqual(c.kind, "custom")
            xy = c.sample(100, np.random.default_rng(1))
            s, eta = c.coordinates(xy)
            self.assertTrue(np.all(np.abs(eta) < 1))

    def test_overlapping_centerline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"path.csv"
            t = np.linspace(0, 2*np.pi, 200)
            np.savetxt(path, np.column_stack((np.cos(t), np.sin(t))), delimiter=",", header="x,y", comments="")
            with self.assertRaises(ValueError):
                Channel(path=str(path))


class EquationTests(unittest.TestCase):
    def test_poiseuille_satisfies_2d_navier_stokes(self):
        class Exact(nn.Module):
            def forward(self, x):
                u = 1-(x[:, 1]/.15)**2
                v = 0*(x[:, 0]**3+x[:, 1]**3)
                p = -2*x[:, 0]/.15**2
                return torch.column_stack((u, v, p))
        x = tensor(np.random.default_rng(1).uniform([0, -.14], [2, .14], (40, 2)), requires_grad=True)
        np.testing.assert_allclose(residuals(Exact(), x, 3.).detach(), 0, atol=2e-5)

    def test_manufactured_3d_solution_satisfies_forced_equations(self):
        class Exact(nn.Module):
            def forward(self, x):
                a, b, c = x.T
                u = torch.column_stack((torch.sin(c)+torch.cos(b), torch.sin(a)+torch.cos(c), torch.sin(b)+torch.cos(a)))
                return torch.column_stack((u, -(u*u).sum(1)/2))
        x = tensor(np.random.default_rng(8).uniform(0, 2*np.pi, (40, 3)), requires_grad=True)
        np.testing.assert_allclose(abc_residual(Exact(), x, 1.).detach(), 0, atol=2e-6)

    def test_architecture_and_zero_slip_force(self):
        c = Channel()
        flow, force = FlowNet(c, 1.), ForceNet(c, 200.)
        hidden_flow = [m for m in flow.net.layers if isinstance(m, nn.Linear)][:-1]
        hidden_force = [m for m in force.net.layers if isinstance(m, nn.Linear)][:-1]
        self.assertEqual([m.out_features for m in hidden_flow], [70]*5)
        self.assertEqual([m.out_features for m in hidden_force], [90]*5)
        x = tensor([[.4, -.3, 10., .01, .1, .2, .4, -.3]])
        np.testing.assert_array_equal(force.force(x, 1.).detach(), np.zeros((1, 2)))

    def test_stokes_relaxation_against_exact_solution(self):
        diameter, re, rho = .02, 1., 1000.
        tau = rho*re*diameter**2/18
        t = np.linspace(0, 6*tau, 100)
        sol = solve_ivp(lambda t, y: stokes_acceleration([1, 0], y, diameter, re, rho),
                        (0, t[-1]), [0, 0], t_eval=t, method="Radau", rtol=1e-9, atol=1e-11)
        self.assertTrue(sol.success)
        np.testing.assert_allclose(sol.y[0], 1-np.exp(-t/tau), atol=2e-9)


class ReferenceTests(unittest.TestCase):
    def test_fem_reproduces_poiseuille_velocity_and_pressure(self):
        c = Channel("straight")
        xy = c.sample(100, np.random.default_rng(1), margin=.1)
        exact = exact_poiseuille(xy, c.half_width)
        for re in (.2, 4.):
            f = solve_flow(c, re, ns=18, nn=6)
            actual = f.evaluate(xy)
            self.assertTrue(f.diagnostics["solver_ok"])
            self.assertLess(relative_error(actual[:, :2], exact[:, :2]), 1e-6)
            self.assertLess(relative_error(actual[:, 2]-actual[:, 2].mean(), exact[:, 2]-exact[:, 2].mean()), 1e-6)

    def test_curved_reference_and_fast_field(self):
        c = Channel("bend")
        xy = c.sample(120, np.random.default_rng(3), margin=.15)
        coarse = solve_flow(c, ns=20, nn=6)
        fine = solve_flow(c, ns=40, nn=10)
        a, b = coarse.evaluate(xy), fine.evaluate(xy)
        self.assertTrue(fine.diagnostics["solver_ok"])
        self.assertLess(relative_error(a[:, :2], b[:, :2]), .04)
        fast = fine.fast_field()(xy)
        self.assertTrue(np.isfinite(fast).all())
        self.assertLess(relative_error(fast[:, :2], b[:, :2]), .02)


class TrajectoryTests(unittest.TestCase):
    def test_outlet_event(self):
        c = Channel("straight")
        f = lambda xy: np.tile([1., 0., 0.], (len(xy), 1))
        for tracer in (False, True):
            t = integrate(c, f, tracer=tracer)
            self.assertEqual(t.outcome, "exited")
            np.testing.assert_allclose(t.state[-1, :2], [c.length, 0], atol=1e-7)

    def test_finite_radius_wall_sticking(self):
        c = Channel("straight")
        f = lambda xy: np.tile([1., .4, 0.], (len(xy), 1))
        d = .02
        t = integrate(c, f, diameter=d)
        self.assertEqual(t.outcome, "stuck")
        self.assertAlmostEqual(t.state[-1, 1], c.half_width-d/2, places=6)
        np.testing.assert_array_equal(t.state[-1, 2:], [0, 0])

    def test_non_finite_field_is_reported_as_failure(self):
        c = Channel("straight")
        def field(xy):
            value = np.tile([1., 0., 0.], (len(xy), 1))
            value[np.asarray(xy)[:, 0] > .1] = np.nan
            return value
        t = integrate(c, field)
        self.assertEqual(t.outcome, "integration_failed")
        self.assertEqual(len(t.t), 1)

    def test_arbitrary_start_and_velocity(self):
        c = Channel("straight")
        f = lambda xy: np.tile([1., 0., 0.], (len(xy), 1))
        t = integrate(c, f, start_xy=[.6, .04], initial_velocity=[0., 0.])
        self.assertEqual(t.outcome, "exited")
        np.testing.assert_allclose(t.state[0], [.6, .04, 0., 0.], atol=1e-10)
        np.testing.assert_allclose(t.state[-1, :2], [2., .04], atol=1e-7)
        with self.assertRaises(ValueError):
            integrate(c, f, start_xy=[.6, 1.])

    def test_evaluation_budget_is_a_recorded_failure(self):
        c = Channel("straight")
        f = lambda xy: np.tile([1., 0., 0.], (len(xy), 1))
        t = integrate(c, f, max_evaluations=2)
        self.assertEqual(t.outcome, "integration_failed")
        self.assertIn("budget", t.message)


class QualityTests(unittest.TestCase):
    def test_successful_solver_does_not_imply_matching_particle_outcomes(self):
        metrics = dict(geometry=dict(width=.3), trajectory=dict(total=2, failures=0,
                       baseline_comparison=[dict(outcome_agreement=False, rms_position_error=.002, common_fraction=.7)]),
                       benchmark3d=dict(velocity_relative_l2=.02, pressure_relative_l2=.13))
        checks = assess(metrics)
        self.assertTrue(checks["trajectory_solver"])
        self.assertFalse(checks["baseline_outcomes"])
        self.assertFalse(checks["baseline_time_coverage"])
        self.assertTrue(checks["abc3d_velocity"])
        self.assertFalse(checks["abc3d_pressure"])


if __name__ == "__main__":
    unittest.main()
