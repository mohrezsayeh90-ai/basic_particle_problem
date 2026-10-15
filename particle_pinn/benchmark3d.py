"""Separate genuinely three-dimensional, forced ABC-flow PINN benchmark.

This exact manufactured solution tests 3-D differential operators and trajectories;
it is explicitly not a solution of the planar channel with finite-depth sidewalls.
"""
import copy
import numpy as np
import torch
from torch import nn
from scipy.integrate import solve_ivp
from .models import MLP, tensor, derivative


def exact_abc(x, re=1.):
    x = np.atleast_2d(x)
    a, b, c = x.T
    u = np.column_stack((np.sin(c)+np.cos(b), np.sin(a)+np.cos(c), np.sin(b)+np.cos(a)))
    return np.column_stack((u, -re*np.sum(u*u, axis=1)/2))


class ABCNet(nn.Module):
    def __init__(self, re=1.):
        super().__init__()
        self.net = MLP(5, 4, 70)
        self.re = re

    def forward(self, xyz):
        params = xyz.new_tensor([self.re, 1.]).expand(len(xyz), -1)
        return self.net(torch.cat((xyz/np.pi-1, params), dim=1))

    def evaluate(self, xyz):
        device = next(self.parameters()).device
        with torch.no_grad():
            return self(tensor(np.asarray(xyz) % (2*np.pi), device)).cpu().numpy()


def abc_residual(model, xyz, re):
    out = model(xyz)
    velocity, pressure = out[:, :3], out[:, 3]
    gradients = [derivative(out[:, j], xyz) for j in range(3)]
    dp = derivative(pressure, xyz)
    forcing = tensor(exact_abc(xyz.detach().cpu().numpy(), re)[:, :3], xyz.device)
    rows = []
    for j in range(3):
        lap = sum(derivative(gradients[j][:, k], xyz)[:, k] for k in range(3))
        rows.append(re*(velocity*gradients[j]).sum(1) + dp[:, j]-lap-forcing[:, j])
    continuity = sum(gradients[j][:, j] for j in range(3))
    return torch.column_stack((continuity, *rows))


def run_benchmark(epochs, seed, device="cpu", lbfgs_steps=200, initial=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ABCNet().to(device)
    if initial is not None:
        model.load_state_dict(initial)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    boundary = rng.uniform(0, 2*np.pi, (300, 3))
    for i in range(len(boundary)):
        boundary[i, i % 3] = (i % 2)*2*np.pi
    bx, by = tensor(boundary, device), tensor(exact_abc(boundary), device)
    # Only velocity is prescribed at the boundary. Pressure has one gauge value.
    gauge = tensor([[0., 0., 0.]], device)
    p0 = float(exact_abc([[0, 0, 0]])[0, 3])
    monitor = tensor(rng.uniform(0, 2*np.pi, (256, 3)), device, True)

    def objective(x):
        r = abc_residual(model, x, 1.)
        bc = (model(bx)[:, :3]-by[:, :3]).square().mean()
        return r.square().mean()+10*bc+(model(gauge)[:, 3]-p0).square().mean()

    best, best_loss, history = copy.deepcopy(model.state_dict()), float(objective(monitor).detach()), []
    for epoch in range(epochs):
        x = tensor(rng.uniform(0, 2*np.pi, (160, 3)), device, True)
        opt.zero_grad(set_to_none=True)
        loss = objective(x)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
        opt.step()
        # Boundary / PDE monitoring, never exact interior labels during training.
        if epoch % 10 == 0 or epoch == epochs-1:
            value = float(objective(monitor).detach())
            history.append(dict(epoch=epoch+1, loss=value))
            if value < best_loss:
                best, best_loss = copy.deepcopy(model.state_dict()), value
    model.load_state_dict(best)
    fixed = tensor(rng.uniform(0, 2*np.pi, (768, 3)), device, True)
    polish = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=lbfgs_steps,
                              history_size=40, line_search_fn="strong_wolfe")
    evaluations = 0

    def closure():
        nonlocal evaluations, best, best_loss
        polish.zero_grad(set_to_none=True)
        fixed.grad = None
        loss = objective(fixed)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite 3D objective.")
        loss.backward()
        evaluations += 1
        if evaluations % 10 == 0:
            value = float(objective(monitor).detach())
            history.append(dict(epoch=epochs+evaluations, loss=value))
            if value < best_loss:
                best, best_loss = copy.deepcopy(model.state_dict()), value
        return loss
    reason = "training_budget"
    if lbfgs_steps:
        try:
            polish.step(closure)
            value = float(objective(monitor).detach())
            if np.isfinite(value) and value < best_loss:
                best, best_loss = copy.deepcopy(model.state_dict()), value
        except (FloatingPointError, RuntimeError) as error:
            reason = f"lbfgs_stopped: {error}"
    model.load_state_dict(best)
    xy = np.random.default_rng(seed+40).uniform(0, 2*np.pi, (1200, 3))
    pred, exact = model.evaluate(xy), exact_abc(xy)
    from .fem import relative_error
    metrics = dict(velocity_relative_l2=relative_error(pred[:, :3], exact[:, :3]),
                   pressure_relative_l2=relative_error(pred[:, 3]-pred[:, 3].mean(), exact[:, 3]-exact[:, 3].mean()),
                   domain="periodic_cube", physics="forced_ABC_flow", training="PDE_and_boundary_only",
                   stop_reason=reason, lbfgs_evaluations=evaluations,
                   geometry_scope="separate_3D_benchmark_not_finite_depth_channel")
    paths = []
    for point in [[1., 1., 1.], [2., 1., 2.], [3., 2., 1.], [1., 3., 2.]]:
        pair = []
        for field in (model.evaluate, exact_abc):
            sol = solve_ivp(lambda t, y: field(y[None, :])[0, :3], (0, 2), point,
                            rtol=1e-6, atol=1e-8, t_eval=np.linspace(0, 2, 180), max_step=.05)
            pair.append(sol.y.T)
        paths.append(pair)
    return model, xy, pred, exact, paths, metrics, history


def plot_benchmark(folder, result):
    from .figures import plt, save, COLORS
    model, xyz, pred, exact, paths, metrics, history = result
    fig = plt.figure(figsize=(12, 5))
    for j, name in enumerate(("3D PINN paths", "ABC reference paths")):
        ax = fig.add_subplot(1, 2, j+1, projection="3d")
        for i, pair in enumerate(paths):
            ax.plot(*pair[j].T, color=COLORS[i], lw=1.6)
        ax.set(title=name, xlabel="x", ylabel="y", zlabel="z")
    save(fig, folder, "19_3d_benchmark")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    error = np.linalg.norm(pred[:, :3]-exact[:, :3], axis=1)
    axes[0].scatter(np.linalg.norm(exact[:, :3], axis=1), np.linalg.norm(pred[:, :3], axis=1), s=3, alpha=.25)
    axes[0].plot([0, 3.5], [0, 3.5], "k--")
    axes[0].set(title="3D velocity validation", xlabel="ABC speed", ylabel="PINN speed")
    axes[1].semilogy([r["epoch"] for r in history], [r["loss"] for r in history])
    axes[1].set(title="3D physics loss", xlabel="Epoch", ylabel="Loss")
    save(fig, folder, "20_3d_validation")
