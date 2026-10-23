"""Animate surrogate preselection in CMA-ES on one 2D BBOB problem.

`record` runs the optimizer once per model and saves per-generation traces to
JSON. `render` turns that JSON into a GIF. The split keeps the slow part (DOE
needs its function bank and VAE) out of the render loop, and the JSON is the
input for a future web player.
"""

import argparse
import json
from pathlib import Path

import numpy as np

TRACE_DIR = Path("data") / "traces"
GIF_DIR = Path("graphs") / "animation"
DIM = 2
# 0.03 cells: about 30 cells across the tightest zoom (MIN_HALF_WIDTH).
GRID_SIZE = 321
MIN_HALF_WIDTH = 0.5
CAMERA_SMOOTHING = 0.3
MODEL_COLORS = {
    "none": "#E04836",
    "nn3": "tab:blue",
    "doe": "tab:purple",
    "oracle": "tab:green",
}


def trace_path(function, instance):
    return TRACE_DIR / f"f{function}_d{DIM}_i{instance}.json"


def bbob_problem(function, instance):
    import cocoex

    suite = cocoex.Suite("bbob", "", f"function_indices:{function} dimensions:{DIM} instance_indices:{instance}")
    return suite.get_problem_by_function_dimension_instance(function, DIM, instance)


def landscape(problem):
    import scipy.optimize

    axis = np.linspace(-5.0, 5.0, GRID_SIZE)
    xx, yy = np.meshgrid(axis, axis)
    values = np.array([problem(np.array(x)) for x in np.column_stack([xx.ravel(), yy.ravel()])]).reshape(xx.shape)
    # cocoex does not expose f_opt or x_opt. In 2D a local polish from the
    # best grid point finds the global optimum to plotting precision.
    start = np.array([xx.ravel()[np.argmin(values)], yy.ravel()[np.argmin(values)]])
    polished = scipy.optimize.minimize(problem, start, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
    return values, polished.x, float(min(polished.fun, values.min()))


def selected_configs():
    import main

    doe_name = "doe_{}_{}".format(*main.load_best_doe_hparams())
    keys = {"no_surrogate": "none", "nn3": "nn3", doe_name: "doe", "oracle": "oracle"}
    configs = {keys[main.config_model_name(c)]: c for c in main.default_configs() if main.config_model_name(c) in keys}
    missing = set(keys.values()) - set(configs)
    if missing:
        raise ValueError(f"default_configs() lacks models: {sorted(missing)}")
    return main, configs


def record_traces(function, instance):
    import evo

    main, configs = selected_configs()
    runs = {}
    for key, (pop_size, gen_mult, dim_red, model) in configs.items():
        assert pop_size is None and dim_red is None, "animation assumes default population and no dim reduction"
        model_f, model_kwargs = (model[0], model[2] if len(model) > 2 else {}) if model else (None, {})
        surrogate = main.build_surrogate(model_f, None, model_kwargs, DIM)
        trace = []
        evo.optimize(
            bbob_problem(function, instance),
            surrogate,
            pop_size=5 * DIM,
            true_evals=main.budget * DIM,
            gen_mult=gen_mult,
            printing=False,
            seed=42,
            trace=trace,
        )
        runs[key] = trace
        print(f"recorded {key}: {len(trace)} generations")

    values, x_opt, f_opt = landscape(bbob_problem(function, instance))
    result = {
        "function": function,
        "function_name": main.func_names[function - 1],
        "instance": instance,
        "dim": DIM,
        "f_opt": f_opt,
        "x_opt": x_opt.tolist(),
        # log10 precision keeps the contour readable across many orders of magnitude.
        "landscape_log10": np.log10(values - f_opt + 1e-8).round(3).tolist(),
        "runs": runs,
    }
    path = trace_path(function, instance)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    print(f"wrote {path}")


def ellipse_params(cov, n_std=2.0):
    eigvals, eigvecs = np.linalg.eigh(np.asarray(cov))
    angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals[::-1], 0.0))
    return width, height, angle


def camera_path(trace, x_opt):
    """Per-generation square view (center x, center y, half width).

    The view covers the 2-sigma box of the search distribution and the
    optimum. Keeping the optimum in view shows how far a stalled run is from
    it. MIN_HALF_WIDTH stops the zoom where the stored landscape grid gets too
    coarse, and smoothing stops the view from jumping when the distribution
    shifts.
    """
    x_opt = np.asarray(x_opt)
    views = []
    center, log_half = np.zeros(2), np.log(5.0)
    for i, generation in enumerate(trace):
        mean = np.asarray(generation["mean"])
        reach = 2.0 * np.sqrt(np.diag(generation["cov"]))
        low = np.minimum(mean - reach, x_opt)
        high = np.maximum(mean + reach, x_opt)
        target_half = np.clip(0.75 * np.max(high - low), MIN_HALF_WIDTH, 5.0)
        target_center = np.clip((low + high) / 2, -5.0 + target_half, 5.0 - target_half)
        if i > 0:
            center += CAMERA_SMOOTHING * (target_center - center)
            log_half += CAMERA_SMOOTHING * (np.log(target_half) - log_half)
        views.append((*center, float(np.exp(log_half))))
    return views


def render_gif(function, instance, fps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Ellipse

    from control_analysis.constants import PRIMARY_DOE_MODEL, display_model_label

    data = json.loads(trace_path(function, instance).read_text())
    runs = data["runs"]
    labels = {"none": "No surrogate", "nn3": display_model_label("nn3"), "doe": display_model_label(PRIMARY_DOE_MODEL), "oracle": "Oracle"}
    f_opt = data["f_opt"]
    frame_count = max(len(trace) for trace in runs.values())
    evals = {key: np.cumsum([len(g["picked"]) for g in trace]) for key, trace in runs.items()}
    regret = {key: np.log10(np.array([g["best"] for g in trace]) - f_opt + 1e-8) for key, trace in runs.items()}

    landscape_log10 = np.array(data["landscape_log10"])
    axis = np.linspace(-5.0, 5.0, landscape_log10.shape[0])
    cell = axis[1] - axis[0]
    views = {key: camera_path(trace, data["x_opt"]) for key, trace in runs.items()}

    def draw_landscape(panel, view):
        cx, cy, half = view
        cols = np.abs(axis - cx) <= half + cell
        rows = np.abs(axis - cy) <= half + cell
        visible = landscape_log10[np.ix_(rows, cols)]
        # Levels come from the visible cells only, so zoomed views keep their
        # contrast. Percentiles drop the few cells next to the optimum, which
        # reach log10(1e-8) and would wash out everything else.
        low, high = np.percentile(visible, [2, 99])
        if panel["landscape"] is not None:
            panel["landscape"].remove()
        # Flat contour bands instead of a smooth image keep the GIF palette
        # small: a smooth gradient dithers and grows the file about six times.
        panel["landscape"] = panel["ax"].contourf(
            axis[cols], axis[rows], np.clip(visible, low, max(high, low + 1e-6)),
            levels=14, cmap="Greys", alpha=0.8, zorder=0,
        )
        panel["ax"].set_xlim(cx - half, cx + half)
        panel["ax"].set_ylim(cy - half, cy + half)

    fig = plt.figure(figsize=(8, 9.5), dpi=80)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.25, wspace=0.08, top=0.93, bottom=0.06)
    fig.suptitle(f"CMA-ES preselection on f{function} {data['function_name']}, 2D", fontsize=13)
    panels = {}
    for i, key in enumerate(runs):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        ax.plot(*data["x_opt"], marker="*", color="gold", markersize=12, markeredgecolor="black", zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(labels[key], color=MODEL_COLORS[key], fontweight="bold")
        color = MODEL_COLORS[key]
        panels[key] = {
            "ax": ax,
            "landscape": None,
            "history": ax.scatter([], [], s=6, color=color, alpha=0.25),
            "rejected": ax.scatter([], [], s=22, facecolors="none", edgecolors="0.3", linewidths=0.8),
            "picked": ax.scatter([], [], s=28, color=color, edgecolors="black", linewidths=0.5, zorder=3),
            "ellipse": ax.add_patch(Ellipse((0, 0), 0, 0, fill=False, color=color, linewidth=1.5)),
            "text": ax.text(0.03, 0.03, "", transform=ax.transAxes, fontsize=9,
                            bbox={"facecolor": "white", "alpha": 0.8, "pad": 2}),
        }

    curve_ax = fig.add_subplot(grid[2, :])
    curve_ax.set_xlabel("True evaluations")
    curve_ax.set_ylabel("log10(best f - f_opt)")
    curve_ax.set_xlim(0, max(e[-1] for e in evals.values()))
    curve_ax.set_ylim(min(r.min() for r in regret.values()) - 0.5, max(r.max() for r in regret.values()) + 0.5)
    curve_ax.grid(alpha=0.3)
    curves = {key: curve_ax.plot([], [], color=MODEL_COLORS[key], label=labels[key], linewidth=2)[0] for key in runs}
    curve_ax.legend(loc="lower left", fontsize=8)

    def draw(frame):
        artists = []
        for key, trace in runs.items():
            step = min(frame, len(trace) - 1)
            generation = trace[step]
            pool = np.array(generation["pool"])
            picked = np.zeros(len(pool), dtype=bool)
            picked[generation["picked"]] = True
            earlier = [np.array(g["pool"])[g["picked"]] for g in trace[:step]]
            panel = panels[key]
            panel["history"].set_offsets(np.vstack(earlier) if earlier else np.empty((0, 2)))
            panel["rejected"].set_offsets(pool[~picked] if (~picked).any() else np.empty((0, 2)))
            panel["picked"].set_offsets(pool[picked])
            width, height, angle = ellipse_params(generation["cov"])
            panel["ellipse"].set_center(generation["mean"])
            panel["ellipse"].set_width(width)
            panel["ellipse"].set_height(height)
            panel["ellipse"].set_angle(angle)
            view = views[key][step]
            draw_landscape(panel, view)
            panel["text"].set_text(
                f"evals {evals[key][step]}   zoom {5.0 / view[2]:.0f}x   f - f_opt {generation['best'] - f_opt:.2e}"
            )
            curves[key].set_data(evals[key][: step + 1], regret[key][: step + 1])
            artists += [panel[name] for name in ("history", "rejected", "picked", "ellipse", "text", "landscape")]
            artists.append(curves[key])
        return artists

    # Hold the final state for two seconds so the result is readable.
    animation = FuncAnimation(fig, draw, frames=frame_count + 2 * fps, blit=False)
    out = GIF_DIR / f"f{function}_d{DIM}_i{instance}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out, writer=PillowWriter(fps=fps))
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["record", "render", "all"])
    parser.add_argument("--function", type=int, default=15, choices=range(1, 25), metavar="1-24")
    parser.add_argument("--instance", type=int, default=1)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()
    if args.command in {"record", "all"}:
        record_traces(args.function, args.instance)
    if args.command in {"render", "all"}:
        render_gif(args.function, args.instance, args.fps)


if __name__ == "__main__":
    main()
