#!/usr/bin/env python3
"""Parses the log written by examples/qp_2d_trajectory_smoothing.cpp and plots
the optimized 2D path, colored by acceleration, against its constraint geometry
(parsed from the log, not hardcoded) plus cost-vs-iteration."""
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch

GEOMETRY_RE = re.compile(
    r"points_per_dim=(\d+) anchors=(.*) corridor=\(([\d.eE+-]+),([\d.eE+-]+),([\d.eE+-]+),([\d.eE+-]+)\)"
)
ANCHOR_RE = re.compile(r"\((\d+),([\d.eE+-]+),([\d.eE+-]+)\)")
LINE_RE = re.compile(
    r"iter=(\d+),cost=([\d.eE+-]+),equality_constraint=[\d.eE+-]+,"
    r"inequality_constraint=[\d.eE+-]+,x=(.+)"
)
ELAPSED_RE = re.compile(r"SOLVE elapsed_ms=([\d.eE+-]+)")


def parse_log(path):
    points_per_dim = None
    anchors = []
    corridor = None
    elapsed_ms = None
    iters, costs, xs, ys = [], [], [], []

    with open(path) as f:
        for line in f:
            m = GEOMETRY_RE.search(line)
            if m:
                points_per_dim = int(m.group(1))
                anchors = [(int(i), float(x), float(y)) for i, x, y in ANCHOR_RE.findall(m.group(2))]
                corridor = tuple(float(v) for v in m.groups()[2:])  # nx, ny, w_left, w_right
                continue
            m = ELAPSED_RE.search(line)
            if m:
                elapsed_ms = float(m.group(1))
                continue
            m = LINE_RE.search(line)
            if m:
                if points_per_dim is None:
                    raise ValueError(f"no GEOMETRY line found in {path} before the first iter= line")
                floats = [float(v) for v in m.group(3).split()]
                # The example's own phase-1 feasibility LP (run internally by QPSolver when no
                # x0 is given) logs through the same logger with one extra scalar (the margin
                # variable t), so its iter= lines are length 2*points_per_dim+1 -- skip those.
                if len(floats) != 2 * points_per_dim:
                    continue
                n = points_per_dim
                iters.append(int(m.group(1)))
                costs.append(float(m.group(2)))
                xs.append(floats[:n])
                ys.append(floats[n:])

    if points_per_dim is None:
        raise ValueError(f"no GEOMETRY line found in {path}")
    if not iters:
        raise ValueError(f"no iter= lines matching points_per_dim={points_per_dim} found in {path}")

    return points_per_dim, anchors, corridor, elapsed_ms, iters, costs, xs, ys


def corridor_strip(anchors, normal, offset_lo, offset_hi, span):
    """Polygon for the band between offset_lo and offset_hi (signed, along normal)
    from the line through the first/last anchor, extended by `span` past both ends."""
    start = anchors[0]
    end = anchors[-1]
    p0 = (start[1], start[2])
    p1 = (end[1], end[2])
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    ux, uy = dx / length, dy / length
    nx, ny = normal
    base0 = (p0[0] - span * ux, p0[1] - span * uy)
    base1 = (p1[0] + span * ux, p1[1] + span * uy)
    return Polygon([
        (base0[0] + offset_lo * nx, base0[1] + offset_lo * ny),
        (base1[0] + offset_lo * nx, base1[1] + offset_lo * ny),
        (base1[0] + offset_hi * nx, base1[1] + offset_hi * ny),
        (base0[0] + offset_hi * nx, base0[1] + offset_hi * ny),
    ])


def acceleration_magnitude(x, y):
    """Discrete 2nd difference at each point (the quantity H actually penalizes),
    padded at the endpoints by repeating the nearest interior value."""
    x, y = np.array(x), np.array(y)
    ax = x[:-2] - 2 * x[1:-1] + x[2:]
    ay = y[:-2] - 2 * y[1:-1] + y[2:]
    accel = np.hypot(ax, ay)
    return np.concatenate([[accel[0]], accel, [accel[-1]]])


def plot(points_per_dim, anchors, corridor, elapsed_ms, iters, costs, xs, ys, out_dir, show):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # iters[0] is the real x0: the phase-1 feasibility point QPSolver computes on its own
    # (no path structure to it -- most waypoints are unconstrained and collapse near the
    # regularizer's origin -- but it's what the solver actually started from, so we show it
    # as-is rather than substituting a nicer-looking but fictitious "guess").
    ax1.scatter(xs[0], ys[0], color="black", marker="x", s=25, alpha=0.7,
                label="Initial Guess (phase-1 feasible point)")

    cmap = cm.get_cmap("coolwarm")
    norm = Normalize(vmin=min(iters), vmax=max(iters))
    for it, x, y in zip(iters, xs, ys):
        if it != iters[0] and it != iters[-1]:
            ax1.plot(x, y, color=cmap(norm(it)), alpha=0.35, linewidth=1)

    final_x, final_y = xs[-1], ys[-1]
    accel = acceleration_magnitude(final_x, final_y)
    points = np.array([final_x, final_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_accel = (accel[:-1] + accel[1:]) / 2
    lc = LineCollection(segments, cmap="plasma", linewidth=2.5, zorder=5)
    lc.set_array(seg_accel)
    ax1.add_collection(lc)
    ax1.scatter(final_x, final_y, c=accel, cmap="plasma", s=10, zorder=6)
    cbar = fig.colorbar(lc, ax=ax1, label="|acceleration| (2nd diff)", fraction=0.046, pad=0.04)

    stable_x = xs[0] + xs[-1] + [a[1] for a in anchors]
    stable_y = ys[0] + ys[-1] + [a[2] for a in anchors]
    margin = 1.0
    ax1.set_xlim(min(stable_x) - margin, max(stable_x) + margin)
    ax1.set_ylim(min(stable_y) - margin, max(stable_y) + margin)

    # Prohibited region: everything outside [-w_right, w_left] along the corridor normal,
    # striped red/white like caution tape. Drawn wide enough to fill past the axis limits.
    nx, ny, w_left, w_right = corridor
    span = 30
    far = 30
    for lo, hi in [(w_left, w_left + far), (-w_right - far, -w_right)]:
        strip = corridor_strip(anchors, (nx, ny), lo, hi, span)
        strip.set(facecolor="white", edgecolor="red", hatch="\\\\\\\\", zorder=0)
        ax1.add_patch(strip)

    markers = {0: ("^", "Start"), points_per_dim - 1: ("s", "End")}
    for idx, ex, ey in anchors:
        marker, _ = markers.get(idx, ("o", "Pass-through"))
        ax1.plot(ex, ey, color="darkgreen", marker=marker, markersize=9, mec="k", zorder=6, ls="")
        ax1.text(ex + 0.2, ey - 0.2, f"k={idx}", fontsize=9, color="darkgreen")

    ax1.set_title("2D trajectory: initial guess vs optimized")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle=":", alpha=0.6)

    trajectory_legend = ax1.legend(handles=[
        Line2D([], [], color="black", marker="x", linestyle="none", label="Initial Guess (phase-1 feasible point)"),
        Line2D([], [], color=cm.get_cmap("plasma")(0.5), label="Optimized Path (colored by |accel|)"),
    ], loc="upper left", fontsize=8)
    ax1.add_artist(trajectory_legend)

    ax1.legend(handles=[
        Line2D([], [], color="darkgreen", marker="^", linestyle="none", mec="k", label="Start (equality)"),
        Line2D([], [], color="darkgreen", marker="s", linestyle="none", mec="k", label="End (equality)"),
        Line2D([], [], color="darkgreen", marker="o", linestyle="none", mec="k", label="Pass-through (equality)"),
        Patch(facecolor="white", edgecolor="red", hatch="\\\\\\\\", label="Prohibited (inequality)"),
    ], loc="lower right", fontsize=8, title="Constraints", title_fontsize=9)

    iters_p1 = [i + 1 for i in iters]  # +1 so iter=0 survives the log axis
    label = f"{iters[-1]} iters"
    if elapsed_ms is not None:
        label += f", {elapsed_ms:.2f} ms"
    ax2.plot(iters_p1, costs, marker="s", color="crimson", linewidth=2, label=label)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("iteration + 1 (log)")
    ax2.set_ylabel("cost (log)")
    ax2.set_title("Convergence")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    x_divider = (cbar.ax.get_position().x1 + ax2.get_position().x0) / 2
    fig.add_artist(plt.Line2D([x_divider, x_divider], [0.03, 0.95],
                              transform=fig.transFigure, color="gray", linewidth=1))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qp_trajectory.png")
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    points_per_dim, anchors, corridor, elapsed_ms, iters, costs, xs, ys = parse_log(args.log)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.log))
    plot(points_per_dim, anchors, corridor, elapsed_ms, iters, costs, xs, ys, out_dir, not args.no_show)


if __name__ == "__main__":
    main()
