"""The two networks in the supplied schematic and their physics losses."""
import copy
import time
import numpy as np
import torch
from torch import nn


def tensor(x, device="cpu", requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, device=device, requires_grad=requires_grad)


class MLP(nn.Module):
    def __init__(self, nin, nout, width, depth=5):
        super().__init__()
        layers = []
        for a, b in zip([nin] + [width]*depth, [width]*depth + [nout]):
            linear = nn.Linear(a, b)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if b == width:
                layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FlowNet(nn.Module):
    """[x,y,R_in,R_out,Re] -> [u,v,p], five hidden layers of 70."""
    def __init__(self, channel, re):
        super().__init__()
        self.net = MLP(5, 3, 70)
        xy, _, _ = channel.grid(100, 3)
        low, high = xy.min(axis=0), xy.max(axis=0)
        self.register_buffer("center", tensor((high+low)/2))
        self.register_buffer("scale", tensor(np.maximum((high-low)/2, channel.half_width)))
        self.register_buffer("flow_parameters", tensor([channel.rin, channel.rout, re]))
        self.pressure_scale = 2*channel.length/channel.half_width**2

    def forward(self, xy):
        # Parameter inputs are present as requested. Each trained model covers its
        # declared geometry/Re case; this does NOT imply parametric generalization.
        params = self.flow_parameters.expand(len(xy), -1)
        inputs = torch.cat(((xy-self.center)/self.scale, params), dim=1)
        raw = self.net(inputs)
        return torch.cat((raw[:, :2], raw[:, 2:3]*self.pressure_scale), dim=1)

    def evaluate(self, xy):
        with torch.no_grad():
            return self(tensor(xy, self.center.device)).cpu().numpy()


def derivative(y, x):
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


def residuals(model, xy, re):
    out = model(xy)
    u, v, p = out[:, 0], out[:, 1], out[:, 2]
    du, dv, dp = derivative(u, xy), derivative(v, xy), derivative(p, xy)
    lap_u = derivative(du[:, 0], xy)[:, 0] + derivative(du[:, 1], xy)[:, 1]
    lap_v = derivative(dv[:, 0], xy)[:, 0] + derivative(dv[:, 1], xy)[:, 1]
    continuity = du[:, 0] + dv[:, 1]
    momentum_x = re*(u*du[:, 0]+v*du[:, 1]) + dp[:, 0]-lap_u
    momentum_y = re*(u*dv[:, 0]+v*dv[:, 1]) + dp[:, 1]-lap_v
    return torch.stack((continuity, momentum_x, momentum_y), dim=1)


def train_flow(channel, re, epochs, samples, seed, device="cpu", initial=None, lbfgs_steps=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = FlowNet(channel, re).to(device)
    if initial is not None:
        model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=5e-5)
    bxy, buv = channel.boundary(max(40, samples//4))
    bx, bv = tensor(bxy, device), tensor(buv, device)
    gauge = tensor(channel.map(np.array([channel.length]), np.array([0.])), device)
    qs, qe = np.meshgrid(np.linspace(.05, .95, 7)*channel.length, np.linspace(-1, 1, 25), indexing="ij")
    qx = tensor(channel.map(qs.ravel(), qe.ravel()), device)
    _, qt, _ = channel.frame(qs[:, 0])
    qt = tensor(qt, device)
    qeta = tensor(np.linspace(-1, 1, 25), device)

    def flux_loss():
        uv = model(qx)[:, :2].reshape(7, 25, 2)
        tangential = (uv*qt[:, None, :]).sum(2)
        normalized_flux = torch.trapezoid(tangential, qeta, dim=1)/(4/3)
        return (normalized_flux-1).square().mean()

    def objective(x):
        r = residuals(model, x, re)
        h = channel.half_width
        physics = (r[:, 0]*h).square().mean() + (r[:, 1:]*h*h).square().mean()
        bc = (model(bx)[:, :2]-bv).square().mean()
        pg = (model(gauge)[:, 2]/model.pressure_scale).square().mean()
        flux = flux_loss()
        return physics+20*bc+pg+5*flux, physics, bc, pg, flux
    # Fixed monitor samples make best-checkpoint selection comparable over epochs.
    monitor = tensor(channel.sample(min(samples, 256), np.random.default_rng(seed+91)), device, True)
    best, best_score, history = copy.deepcopy(model.state_dict()), float(objective(monitor)[0].detach()), []
    started, reason = time.time(), "epoch_budget"
    for epoch in range(epochs):
        xy = tensor(channel.sample(samples, rng), device, True)
        optimizer.zero_grad(set_to_none=True)
        loss, physics, bc, pressure_gauge, flux = objective(xy)
        if not torch.isfinite(loss):
            reason = "non_finite_loss"
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs-1:
            score = float(objective(monitor)[0].detach())
            history.append(dict(epoch=epoch+1, total=float(loss.detach()), physics=float(physics.detach()),
                                boundary=float(bc.detach()), gauge=float(pressure_gauge.detach()),
                                flux=float(flux.detach()), monitor=score, optimizer="Adam"))
            if np.isfinite(score) and score < best_score:
                best, best_score = copy.deepcopy(model.state_dict()), score
            if epoch % max(epochs//5, 10) == 0:
                print(f"Flow {channel.kind}: {epoch+1}/{epochs}, loss={float(loss.detach()):.4g}", flush=True)
    model.load_state_dict(best)
    polish_evaluations = 0
    if lbfgs_steps and reason != "non_finite_loss":
        fixed = tensor(channel.sample(max(samples, 512), rng), device, True)
        polish = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=lbfgs_steps,
                                  history_size=40, line_search_fn="strong_wolfe")

        def closure():
            nonlocal best, best_score, polish_evaluations
            polish.zero_grad(set_to_none=True)
            fixed.grad = None
            loss, physics, bc, pg, flux = objective(fixed)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite L-BFGS objective; restoring best finite checkpoint.")
            loss.backward()
            polish_evaluations += 1
            if polish_evaluations % 10 == 0:
                score = float(objective(monitor)[0].detach())
                history.append(dict(epoch=epochs+polish_evaluations, total=float(loss.detach()),
                                    physics=float(physics.detach()), boundary=float(bc.detach()),
                                    gauge=float(pg.detach()), flux=float(flux.detach()), monitor=score, optimizer="LBFGS"))
                if score < best_score:
                    best, best_score = copy.deepcopy(model.state_dict()), score
            return loss
        try:
            polish.step(closure)
            score = float(objective(monitor)[0].detach())
            if np.isfinite(score) and score < best_score:
                best, best_score = copy.deepcopy(model.state_dict()), score
        except (FloatingPointError, RuntimeError) as error:
            reason = f"lbfgs_stopped: {error}"
        model.load_state_dict(best)
    model.eval()
    return model, history, dict(stop_reason=reason, best_monitor=best_score, seconds=time.time()-started,
                               epochs_completed=epoch+1 if epochs else 0, lbfgs_evaluations=polish_evaluations,
                               training="physics_boundary_and_integral_mass_conservation")


class ForceNet(nn.Module):
    """[u,v,p,d_p,x_p,y_p,u_p,v_p] -> [F_x,F_y], 5 x 90.

    Learns the normalized creeping-flow drag from its algebraic physics residual.
    Force is per displaced-fluid mass; acceleration is force / density_ratio.
    Subtracting the zero-slip network output enforces exactly zero drag at rest
    relative to the fluid, without replacing the learned slip response.
    """
    def __init__(self, channel, pressure_scale):
        super().__init__()
        self.net = MLP(8, 2, 90)
        low, high = channel.curve(channel.lookup_s).min(0), channel.curve(channel.lookup_s).max(0)
        self.register_buffer("offset", tensor([0, 0, 0, 0, *((low+high)/2), 0, 0]))
        self.register_buffer("scale", tensor([1, 1, pressure_scale, channel.half_width,
                                               max(high[0]-low[0], channel.half_width),
                                               max(high[1]-low[1], channel.half_width), 1, 1]))

    def normalized_drag(self, inputs):
        z = (inputs-self.offset)/self.scale
        zero = z.clone()
        zero[:, 6:8] = z[:, :2]
        return self.net(z)-self.net(zero)

    def force(self, inputs, re):
        return 18/(re*inputs[:, 3:4]**2)*self.normalized_drag(inputs)

    def evaluate(self, flow, xy, velocity, diameter, re):
        inputs = np.r_[flow[:3], diameter, xy[:2], velocity[:2]][None, :]
        with torch.no_grad():
            return self.force(tensor(inputs, self.offset.device), re).cpu().numpy()[0]


def train_force(channel, epochs, seed, device="cpu"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pressure_scale = 2*channel.length/channel.half_width**2
    model = ForceNet(channel, pressure_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=8e-4)

    def batch(n):
        flow = rng.uniform(-1.3, 1.3, (n, 3))
        flow[:, 2] *= pressure_scale
        diameter = rng.uniform(.04, .4, (n, 1))*channel.half_width
        xy = channel.sample(n, rng)
        vp = rng.uniform(-1.3, 1.3, (n, 2))
        x = np.column_stack((flow, diameter, xy, vp))
        return tensor(x, device), tensor(flow[:, :2]-vp, device)

    xval, yval = batch(1024)
    best, best_loss, history = copy.deepcopy(model.state_dict()), float("inf"), []
    reason = "epoch_budget"
    for epoch in range(epochs):
        x, y = batch(512)
        opt.zero_grad(set_to_none=True)
        loss = (model.normalized_drag(x)-y).square().mean()
        if not torch.isfinite(loss):
            reason = "non_finite_loss"
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
        opt.step()
        if epoch % 10 == 0 or epoch == epochs-1:
            with torch.no_grad():
                val = float((model.normalized_drag(xval)-yval).square().mean())
            history.append(dict(epoch=epoch+1, train=float(loss.detach()), validation=val))
            if val < best_loss:
                best_loss, best = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    xtest, ytest = batch(2048)
    with torch.no_grad():
        rel = float(torch.linalg.norm(model.normalized_drag(xtest)-ytest)/torch.linalg.norm(ytest))
    return model.eval(), history, dict(heldout_relative_l2=rel, stop_reason=reason,
                                      physics="Stokes_drag", training="algebraic_physics_residual")


def flow_diagnostics(model, channel, re, seed):
    xy = tensor(channel.sample(512, np.random.default_rng(seed)), model.center.device, True)
    r = residuals(model, xy, re).detach().cpu().numpy()
    b, bv = channel.boundary(80)
    bpred = model.evaluate(b)[:, :2]
    flux = []
    eta = np.linspace(-1, 1, 101)
    for s in np.linspace(0, channel.length, 25):
        _, t, _ = channel.frame(s)
        vel = model.evaluate(channel.map(np.full(101, s), eta))[:, :2]
        flux.append(np.trapezoid(vel@t, eta)*channel.half_width)
    return dict(continuity_rms=float(np.sqrt(np.mean(r[:, 0]**2))),
                momentum_rms=float(np.sqrt(np.mean(r[:, 1:]**2))),
                boundary_rms=float(np.sqrt(np.mean((bpred-bv)**2))),
                flux_relative_deviation=float(np.max(np.abs(np.array(flux)/(4*channel.half_width/3)-1))),
                flux=flux)
