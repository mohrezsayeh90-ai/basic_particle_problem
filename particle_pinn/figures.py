"""Publication exports. All panels identify the actual data source."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/particle-pinn-matplotlib")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 240, "figure.dpi": 110,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
COLORS = plt.get_cmap("tab10").colors


def save(fig, folder, name):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def walls(ax, channel, three=False, z=0):
    s = np.linspace(0, channel.length, 500)
    for eta in (-1, 1):
        xy = channel.map(s, np.full(len(s), eta))
        if three:
            ax.plot(xy[:, 0], xy[:, 1], np.full(len(s), z), color=".35", lw=.8)
        else:
            ax.plot(*xy.T, color=".25", lw=1)
    if not three:
        ax.set_aspect("equal")
        ax.set_xlabel("x / L₀")
        ax.set_ylabel("y / L₀")


def architecture(folder):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    specs = [(axes[0], ["x", "y", "Rᵢₙ", "Rₒᵤₜ", "Re"], ["u", "v", "p"], 70, "Flow network"),
             (axes[1], ["u", "v", "p", "dₚ", "xₚ", "yₚ", "uₚ", "vₚ"], ["Fₓ", "Fᵧ"], 90, "Force network")]
    for ax, inputs, outputs, width, title in specs:
        layers = [inputs] + [["", "", "⋮", "", ""]]*5 + [outputs]
        locations = []
        for i, labels in enumerate(layers):
            ys = np.linspace(4.5, .5, len(labels))
            locations.append([(i, y) for y in ys])
        for left, right in zip(locations[:-1], locations[1:]):
            ax.add_collection(LineCollection([[a, b] for a in left for b in right], colors=".6", linewidths=.35, alpha=.55))
        for i, (labels, positions) in enumerate(zip(layers, locations)):
            for text, (x, y) in zip(labels, positions):
                if text == "⋮":
                    ax.text(x, y, text, ha="center", va="center", fontsize=20)
                else:
                    ax.add_patch(Circle((x, y), .14, color="#087dac" if 0<i<6 else "#dcecf4", ec=".2", zorder=3))
                    ax.text(x, y, text, ha="center", va="center", zorder=4, fontsize=9)
            if 0<i<6:
                ax.text(i, 4.95, str(width), ha="center")
        ax.set(xlim=(-.35, 6.35), ylim=(0, 5.25), title=title, aspect="equal")
        ax.axis("off")
    fig.text(.5, .01, "Five tanh hidden layers per network · nondimensional inputs · velocity-to-force coupling", ha="center")
    fig.savefig(folder/"02_architecture.svg", bbox_inches="tight")
    save(fig, folder, "02_architecture")


def geometry(folder, channel, seed):
    fig, ax = plt.subplots(figsize=(8, 4))
    walls(ax, channel)
    xy = channel.sample(600, np.random.default_rng(seed))
    ax.scatter(*xy.T, s=2, c="#3896c4", alpha=.45, label="Collocation")
    b, _ = channel.boundary(40)
    ax.scatter(*b.T, s=5, c="#d56c29", label="Boundary")
    ax.set_title("Channel and sampling")
    ax.legend(loc="best", markerscale=2)
    save(fig, folder, "01_geometry")


def losses(folder, flow_history, force_history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("total", "physics", "boundary", "monitor"):
        axes[0].semilogy([r["epoch"] for r in flow_history], [max(r[key], 1e-15) for r in flow_history], label=key)
    for key in ("train", "validation"):
        axes[1].semilogy([r["epoch"] for r in force_history], [max(r[key], 1e-15) for r in force_history], label=key)
    for ax, title in zip(axes, ("Flow loss", "Drag loss")):
        ax.set(title=title, xlabel="Training step", ylabel="Loss")
        ax.legend()
    save(fig, folder, "03_losses")


def field_panels(folder, channel, points, pred, reference, shape=None):
    speed = lambda x: np.linalg.norm(x[:, :2], axis=1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    arrays = [speed(pred), speed(reference), np.linalg.norm(pred[:, :2]-reference[:, :2], axis=1),
              pred[:, 2]-pred[:, 2].mean(), reference[:, 2]-reference[:, 2].mean(),
              np.abs((pred[:, 2]-pred[:, 2].mean())-(reference[:, 2]-reference[:, 2].mean()))]
    titles = ["PINN speed", "FEM speed", "Velocity error", "PINN pressure", "FEM pressure", "Pressure error"]
    for i, (ax, values, title) in enumerate(zip(axes.flat, arrays, titles)):
        limits = {}
        if i in (0, 1):
            limits = dict(vmin=0, vmax=max(np.max(arrays[0]), np.max(arrays[1])))
        if i in (3, 4):
            lim = max(np.max(np.abs(arrays[3])), np.max(np.abs(arrays[4])))
            limits = dict(vmin=-lim, vmax=lim)
        cmap = "coolwarm" if i in (3, 4) else "viridis"
        if shape:
            artist = ax.pcolormesh(points[:, 0].reshape(shape), points[:, 1].reshape(shape),
                                   values.reshape(shape), shading="gouraud", cmap=cmap, **limits)
        else:
            artist = ax.scatter(*points.T, c=values, s=5, cmap=cmap, **limits)
        walls(ax, channel)
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, shrink=.7)
    save(fig, folder, "04_flow_fields")


def parity(folder, pred, ref):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for j, ax in enumerate(axes):
        a, b = ref[:, j].copy(), pred[:, j].copy()
        if j == 2:
            a -= a.mean()
            b -= b.mean()
        ax.scatter(a, b, s=4, alpha=.3)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(title=("Axial velocity", "Transverse velocity", "Pressure")[j], xlabel="FEM", ylabel="PINN")
    save(fig, folder, "05_validation_parity")


def profiles(folder, channel, field, femfield):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    eta = np.linspace(-.98, .98, 100)
    for fraction, ax in zip((.2, .5, .8), axes):
        s = fraction*channel.length
        xy = channel.map(np.full(len(eta), s), eta)
        _, t, _ = channel.frame(s)
        ax.plot(field(xy)[:, :2]@t, eta, label="PINN")
        ax.plot(femfield(xy)[:, :2]@t, eta, "--", label="FEM")
        ax.set(title=f"s / L = {fraction}", xlabel="Tangential velocity", ylabel="n / h")
        ax.legend()
    save(fig, folder, "06_velocity_profiles")


def flux(folder, channel, field, femfield):
    s = np.linspace(0, channel.length, 45)
    eta = np.linspace(-.999, .999, 150)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for f, name in ((field, "PINN"), (femfield, "FEM")):
        q = []
        for a in s:
            _, t, _ = channel.frame(a)
            u = f(channel.map(np.full(len(eta), a), eta))[:, :2]@t
            q.append(np.trapezoid(u, eta)*channel.half_width)
        ax.plot(s/channel.length, np.array(q)/(4*channel.half_width/3), label=name)
    ax.axhline(1, ls=":", c="k", label="Prescribed flux")
    ax.set(title="Mass conservation", xlabel="s / L", ylabel="Q / Q₀")
    ax.legend()
    save(fig, folder, "07_mass_flux")


def refinement(folder, rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].loglog([r["triangles"] for r in rows], [max(r["velocity_error_to_finest"], 1e-15) for r in rows], "o-")
    axes[0].set(title="Mesh sensitivity", xlabel="Elements", ylabel="Velocity difference to finest")
    axes[1].loglog([r["triangles"] for r in rows], [max(r["divergence_l2"], 1e-15) for r in rows], "o-")
    axes[1].set(title="FEM continuity", xlabel="Elements", ylabel="‖∇ · u‖₂")
    save(fig, folder, "08_mesh_sensitivity")


def track_panels(folder, channel, tracks, name, title, group_key="source"):
    groups = list(dict.fromkeys(getattr(t, group_key) for t in tracks))
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(6*len(groups), 4), squeeze=False)
    for ax, group in zip(axes[0], groups):
        walls(ax, channel)
        for i, t in enumerate([t for t in tracks if getattr(t, group_key) == group]):
            xy = t.state[:, :2]
            ax.plot(*xy.T, c=COLORS[i % len(COLORS)], lw=1.4)
            ax.scatter(*xy[0], s=12, c=[COLORS[i % len(COLORS)]])
            ax.scatter(*xy[-1], s=25, marker="x" if t.outcome == "stuck" else "o", c=[COLORS[i % len(COLORS)]])
        ax.set_title(str(group))
    fig.suptitle(title)
    save(fig, folder, name)


def parameter_study(folder, channel, tracks, parameter, name):
    values = sorted(set(getattr(t, parameter) for t in tracks))
    if not values:
        return
    fig, axes = plt.subplots(2, len(values), figsize=(4.5*len(values), 7), squeeze=False)
    sources = list(dict.fromkeys(t.source for t in tracks))[:2]
    for row, source in enumerate(sources):
        for ax, value in zip(axes[row], values):
            walls(ax, channel)
            selected = [t for t in tracks if t.source == source and getattr(t, parameter) == value]
            for i, t in enumerate(selected):
                ax.plot(*t.state[:, :2].T, color=COLORS[i%10], lw=1.2)
                if t.outcome == "stuck":
                    ax.scatter(*t.state[-1, :2], marker="x", c=[COLORS[i%10]])
            ax.set_title(f"{source}\n{'dₚ' if parameter=='diameter' else 'ρₚ / ρf'} = {value:g}")
    save(fig, folder, name)


def outcomes(folder, tracks):
    groups = list(dict.fromkeys((t.source, t.case) for t in tracks))
    fig, ax = plt.subplots(figsize=(max(8, len(groups)*.55), 4.5))
    bottom = np.zeros(len(groups))
    for outcome, color in zip(("exited", "stuck", "time_limit", "backflow_exit", "integration_failed"),
                              ("#3b9c75", "#dd8847", "#5b8dbd", "#956ca5", "#b64d54")):
        values = [sum(t.outcome == outcome for t in tracks if (t.source, t.case)==group)/
                  sum((t.source, t.case)==group for t in tracks) for group in groups]
        ax.bar(range(len(groups)), values, bottom=bottom, color=color, label=outcome.replace("_", " "))
        bottom += np.array(values)
    ax.set_xticks(range(len(groups)), [f"{a}\n{b}" for a, b in groups], rotation=65, ha="right", fontsize=7)
    ax.set(title="Particle outcomes", ylabel="Fraction", ylim=(0, 1.05))
    ax.legend(ncol=3, fontsize=8)
    save(fig, folder, "13_particle_outcomes")


def three_dimensional(folder, channel, tracks):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection="3d")
    depth = 2*channel.half_width
    s = np.linspace(0, channel.length, 160)
    for side in (-1, 1):
        wall = channel.map(s, np.full(len(s), side))
        z = np.array([0., depth])
        ax.plot_surface(np.tile(wall[:, 0, None], (1, 2)), np.tile(wall[:, 1, None], (1, 2)),
                        np.tile(z, (len(s), 1)), color="#71b8db", alpha=.15, linewidth=0)
    for i, t in enumerate(tracks):
        z = depth*(.15+.7*i/max(len(tracks)-1, 1))
        ax.plot(t.state[:, 0], t.state[:, 1], np.full(len(t.t), z), c=COLORS[i%10], lw=1.5)
    ax.set(title="Extruded planar paths", xlabel="x / L₀", ylabel="y / L₀", zlabel="Display depth / L₀")
    ax.view_init(elev=28, azim=-65)
    fig.text(.5, .02, "Planar solution extruded for display; depth offsets are illustrative.", ha="center", fontsize=9)
    save(fig, folder, "14_extruded_paths")


def drag_validation(folder, prediction, target, times, numerical, analytical):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(target.ravel(), prediction.ravel(), s=3, alpha=.2)
    lo, hi = min(target.min(), prediction.min()), max(target.max(), prediction.max())
    axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set(title="Drag validation", xlabel="Stokes normalized force", ylabel="ForceNet normalized force")
    axes[1].plot(times, analytical, "k--", label="Analytical Stokes")
    axes[1].plot(times, numerical, label="ODE solver")
    axes[1].set(title="Velocity relaxation", xlabel="Time / T₀", ylabel="Particle velocity / U₀")
    axes[1].legend()
    save(fig, folder, "15_drag_validation")


def poiseuille(folder, eta, exact, fem, pinn):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(exact, eta, "k--", label="Analytical")
    axes[0].plot(fem, eta, label="FEM")
    axes[0].plot(pinn, eta, label="PINN")
    axes[0].set(title="Poiseuille flow", xlabel="Velocity / U₀", ylabel="y / h")
    axes[0].legend()
    axes[1].semilogx(np.maximum(abs(fem-exact), 1e-15), eta, label="FEM")
    axes[1].semilogx(np.maximum(abs(pinn-exact), 1e-15), eta, label="PINN")
    axes[1].set(title="Profile error", xlabel="Absolute velocity error", ylabel="y / h")
    axes[1].legend()
    save(fig, folder, "16_poiseuille_validation")


def motion_animation(folder, channel, tracks, frames=70):
    sources = list(dict.fromkeys(t.source for t in tracks))[:2]
    if not sources:
        return
    fig, axes = plt.subplots(1, len(sources), figsize=(6*len(sources), 4), squeeze=False)
    moving = []
    end = max(t.t[-1] for t in tracks)
    for ax, source in zip(axes[0], sources):
        walls(ax, channel)
        ax.set_title(source)
        for i, track in enumerate([t for t in tracks if t.source == source]):
            ax.plot(*track.state[:, :2].T, c=COLORS[i%10], alpha=.2, lw=.8)
            line, = ax.plot([], [], "o", c=COLORS[i%10], ms=5)
            moving.append((line, track))
    clock = fig.text(.5, .02, "", ha="center")

    def update(frame):
        time = end*frame/max(frames-1, 1)
        for marker, track in moving:
            x = np.interp(time, track.t, track.state[:, 0])
            y = np.interp(time, track.t, track.state[:, 1])
            marker.set_data([x], [y])
            marker.set_visible(time <= track.t[-1] or track.outcome == "stuck")
        clock.set_text(f"t / T₀ = {time:.2f}")
        return [x[0] for x in moving]+[clock]
    movie = animation.FuncAnimation(fig, update, frames=frames, interval=70, blit=False)
    movie.save(folder/"17_particle_motion.gif", writer="pillow", fps=14)
    plt.close(fig)


def ensemble(folder, channel, tracks_by_seed):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed, xy, predictions, reference, loss_history in tracks_by_seed:
        errors = np.linalg.norm(predictions[:, :2]-reference[:, :2], axis=1)
        axes[0].plot([r["epoch"] for r in loss_history], [r["monitor"] for r in loss_history], label=f"Seed {seed}")
        axes[1].hist(errors, bins=35, histtype="step", density=True, label=f"Seed {seed}")
    axes[0].set(yscale="log", title="Seed sensitivity", xlabel="Training step", ylabel="Monitor loss")
    axes[1].set(title="Velocity error distribution", xlabel="‖uPINN − uFEM‖", ylabel="Density")
    for ax in axes:
        ax.legend()
    save(fig, folder, "18_seed_sensitivity")


def particle_dynamics(folder, rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    sources = list(dict.fromkeys(row["source"] for row in rows))
    for source in sources:
        selected = [r for r in rows if r["source"] == source]
        times = [r["t"] for r in selected]
        axes[0].plot(times, [np.hypot(r["up"], r["vp"]) for r in selected], label=source)
        axes[1].plot(times, [r["Fx"] for r in selected], label=source)
        axes[2].plot(times, [r["Fy"] for r in selected], label=source)
    for ax, title, ylabel in zip(axes, ("Particle speed", "Axial drag", "Transverse drag"), ("Speed / U₀", "Fₓ*", "Fᵧ*")):
        ax.set(title=title, xlabel="Time / T₀", ylabel=ylabel)
        ax.legend(fontsize=8)
    save(fig, folder, "21_particle_dynamics")


def deposition(folder, channel, tracks):
    sources = ["PINN + ForceNet", "FEM + Stokes"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, source in zip(axes, sources):
        selected = [t for t in tracks if t.source==source and t.case.startswith("diameter")]
        for outcome, marker in (("stuck", "x"), ("exited", "o"), ("time_limit", "s"), ("backflow_exit", "<"), ("integration_failed", "+")):
            group = [t for t in selected if t.outcome==outcome]
            if group:
                ax.scatter([t.seed_eta for t in group], [t.diameter/(2*channel.half_width) for t in group],
                           marker=marker, s=55, label=outcome.replace("_", " "))
        ax.set(title=source, xlabel="Inlet n / h", ylabel="dₚ / channel width")
        if selected:
            ax.legend(fontsize=8)
    save(fig, folder, "22_deposition_map")
