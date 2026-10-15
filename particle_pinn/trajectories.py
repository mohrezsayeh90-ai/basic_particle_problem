"""Lagrangian integration with geometric wall events and transparent failures."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Track:
    source: str
    case: str
    seed_eta: float
    diameter: float
    density_ratio: float
    t: np.ndarray
    state: np.ndarray
    outcome: str
    message: str
    max_particle_re: float


def stokes_acceleration(fluid_velocity, particle_velocity, diameter, re, density_ratio):
    return 18*(np.asarray(fluid_velocity)-np.asarray(particle_velocity))/(density_ratio*re*diameter**2)


def integrate(channel, field, re=1., diameter=.02, density_ratio=1000., seed_eta=0.,
              force_model=None, tracer=False, source="FEM + Stokes", case="baseline",
              tmax=None, samples=260, max_evaluations=16000, start_xy=None, initial_velocity=None):
    if re <= 0 or diameter <= 0 or density_ratio <= 0:
        raise ValueError("Re, diameter, and density ratio must be positive.")
    radius = 0. if tracer else diameter/2
    if start_xy is None:
        start = channel.map(np.array([1e-5*channel.length]), np.array([seed_eta]))[0]
    else:
        start = np.asarray(start_xy, dtype=float)
        if start.shape != (2,) or not np.all(np.isfinite(start)):
            raise ValueError("start_xy must contain two finite planar coordinates.")
        s0, eta0 = channel.coordinates(start[None, :])
        if not 0 < s0[0] < channel.length:
            raise ValueError("Start point must be strictly inside the channel end caps.")
        seed_eta = float(eta0[0])
    if radius >= channel.half_width or abs(seed_eta) >= 1-radius/channel.half_width:
        raise ValueError("Particle seed intersects a wall.")
    initial_flow = np.asarray(field(start[None, :]))[0]
    velocity = initial_flow[:2] if initial_velocity is None or tracer else np.asarray(initial_velocity, dtype=float)
    if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
        raise ValueError("Initial velocity must have two finite components.")
    y0 = np.r_[start, velocity]
    tmax = tmax if tmax is not None else 4.5*channel.length
    evaluations = 0

    def rhs(t, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_evaluations:
            raise RuntimeError("Trajectory RHS evaluation budget exhausted.")
        flow = np.asarray(field(y[:2][None, :]))[0]
        if not np.all(np.isfinite(flow)):
            raise FloatingPointError("Flow prediction is non-finite along this trajectory.")
        if tracer:
            return np.r_[flow[:2], 0., 0.]
        if force_model is None:
            acc = stokes_acceleration(flow[:2], y[2:], diameter, re, density_ratio)
        else:
            acc = force_model.evaluate(flow, y[:2], y[2:], diameter, re)/density_ratio
        return np.r_[y[2:], acc]

    def wall(t, y):
        _, eta = channel.coordinates(y[:2][None, :])
        return channel.half_width*(1-abs(eta[0]))-radius

    def outlet(t, y):
        s, _ = channel.coordinates(y[:2][None, :])
        return channel.length-s[0]

    def inlet(t, y):
        s, _ = channel.coordinates(y[:2][None, :])
        return s[0]

    wall.terminal = outlet.terminal = inlet.terminal = True
    wall.direction = outlet.direction = inlet.direction = -1
    try:
        # Resolve motion relative to the wall spacing to avoid jumping through
        # both sides between event evaluations; Radau handles short Stokes times.
        max_step = min(.06, channel.half_width/10)
        tau = density_ratio*re*diameter**2/18
        method = "RK45" if tracer or tau >= max_step/3 else "Radau"
        sol = solve_ivp(rhs, (0, tmax), y0, method=method,
                        rtol=2e-6, atol=2e-8, dense_output=True,
                        max_step=max_step, events=(wall, outlet, inlet))
        end = float(sol.t[-1])
        times = np.linspace(0, end, samples)
        state = sol.sol(times).T
        if len(sol.t_events[0]):
            outcome = "stuck"
            if not tracer:
                state[-1, 2:] = 0.
        elif len(sol.t_events[1]):
            outcome = "exited"
        elif len(sol.t_events[2]):
            outcome = "backflow_exit"
        else:
            outcome = "time_limit" if sol.success else "integration_failed"
        flow = np.asarray(field(state[:, :2]))[:, :2]
        if tracer:
            state[:, 2:] = flow
        slip = np.linalg.norm(flow-state[:, 2:], axis=1) if not tracer else np.zeros(samples)
        max_rep = float(np.max(re*diameter*slip))
        return Track(source, case, seed_eta, diameter, density_ratio, times, state, outcome,
                     sol.message, max_rep)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        # A failed numerical path stays a failed, one-point path, never fabricated.
        return Track(source, case, seed_eta, diameter, density_ratio, np.array([0.]),
                     y0[None, :], "integration_failed", str(error), float("nan"))


def track_error(a, b):
    """Compare only the common physical time interval; report its coverage."""
    end = min(a.t[-1], b.t[-1])
    if end <= 0:
        return dict(rms_position_error=None, common_time=0., common_fraction=0.,
                    outcome_agreement=a.outcome == b.outcome)
    times = np.linspace(0, end, 150)
    pa = np.column_stack([np.interp(times, a.t, a.state[:, i]) for i in range(2)])
    pb = np.column_stack([np.interp(times, b.t, b.state[:, i]) for i in range(2)])
    return dict(rms_position_error=float(np.sqrt(np.mean(np.sum((pa-pb)**2, axis=1)))),
                common_time=float(end), common_fraction=float(end/max(a.t[-1], b.t[-1])),
                outcome_agreement=a.outcome == b.outcome)
