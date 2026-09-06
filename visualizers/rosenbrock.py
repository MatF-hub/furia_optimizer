#!/usr/bin/env python3
"""Parses solver_log.log (written by examples/rosenbrock.cpp) and plots the
GradientDescent/BFGS/ExactNewton/GaussNewton paths on the Rosenbrock surface
plus cost-vs-iteration."""
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

METHOD_RE = re.compile(r"=== METHOD (\S+) ===")
LINE_RE = re.compile(r"iter=(\d+),cost=([\d.eE+-]+),grad_norm=[\d.eE+-]+,x=(.+)")
TIME_RE = re.compile(r"METHOD (\S+) elapsed_ms=([\d.eE+-]+)")


def parse_log(path):
    methods = {}
    current = None
    with open(path) as f:
        for line in f:
            m = TIME_RE.search(line)
            if m:
                methods[m.group(1)]["elapsed_ms"] = float(m.group(2))
                continue
            m = METHOD_RE.search(line)
            if m:
                current = m.group(1)
                methods[current] = {"iter": [], "cost": [], "x0": [], "x1": [], "elapsed_ms": 0.0}
                continue
            m = LINE_RE.search(line)
            if m and current:
                x0, x1 = (float(v) for v in m.group(3).split())
                d = methods[current]
                d["iter"].append(int(m.group(1)))
                d["cost"].append(float(m.group(2)))
                d["x0"].append(x0)
                d["x1"].append(x1)
    return methods


def rosenbrock(x, y, a=1.0, b=100.0):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def plot(methods, out_dir, show):
    fig = plt.figure(figsize=(12, 5))
    colors = {"GradientDescent": "red", "BFGS": "orange", "ExactNewton": "cyan", "GaussNewton": "purple"}

    linestyles = {"GradientDescent": "-", "BFGS": "--", "ExactNewton": ":", "GaussNewton": "-."}

    all_x0 = [v for d in methods.values() for v in d["x0"]]
    all_x1 = [v for d in methods.values() for v in d["x1"]]
    margin = 0.2
    x_lo, x_hi = min(all_x0) - margin, max(all_x0) + margin
    y_lo, y_hi = min(all_x1) - margin, max(all_x1) + margin

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    x = np.linspace(x_lo, x_hi, 200)
    y = np.linspace(y_lo, y_hi, 200)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, rosenbrock(X, Y), cmap="viridis", norm=LogNorm(), alpha=0.6, linewidth=0)
    for name, d in methods.items():
        z = rosenbrock(np.array(d["x0"]), np.array(d["x1"]))
        ax1.plot(d["x0"], d["x1"], z, color=colors.get(name, "black"),
                  linestyle=linestyles.get(name, "-"), label=name, linewidth=2.5)
        ax1.scatter(d["x0"][0], d["x1"][0], z[0], color="green", s=40)
        ax1.scatter(d["x0"][-1], d["x1"][-1], z[-1], color="black", marker="*", s=80)
    ax1.view_init(elev=35, azim=-120)
    ax1.set_xlabel("x0")
    ax1.set_ylabel("x1")
    ax1.set_zlabel("cost")
    ax1.legend()
    ax1.set_title("Optimizer paths")

    ax2 = fig.add_subplot(1, 2, 2)
    for name, d in methods.items():
        iters = [i + 1 for i in d["iter"]]  # +1 so iter=0 survives the log axis
        label = f"{name} ({d['iter'][-1]} iters, {d['elapsed_ms']:.2f} ms)"
        ax2.plot(iters, d["cost"], color=colors.get(name, "black"), label=label)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("iteration + 1 (log)")
    ax2.set_ylabel("cost (log)")
    ax2.legend()
    ax2.set_title("Convergence")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rosenbrock.png")
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

    out_dir = args.out or os.path.dirname(os.path.abspath(args.log))
    plot(parse_log(args.log), out_dir, not args.no_show)


if __name__ == "__main__":
    main()
