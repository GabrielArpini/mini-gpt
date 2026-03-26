"""
Plots scaling law curves from scaling_results.json.
Run after scaling_laws.py finishes (or after any subset of scales).

Usage:
    python scripts/plot_scaling.py
"""
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


RESULTS_PATH = "scaling_results.json"


def power_law(x, a, b):
    """L = a * x^b  (linear on log-log axes)"""
    return a * np.power(x, b)


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def main():
    results = load_results()
    if len(results) < 2:
        print("Need at least 2 completed scales to plot. Run more scales first.")
        sys.exit(1)

    names = [r["name"] for r in results]
    params = np.array([r["n_params"] for r in results], dtype=np.float64)
    tokens = np.array([r["tokens_trained"] for r in results], dtype=np.float64)
    flops = np.array([r["total_flops"] for r in results], dtype=np.float64)
    losses = np.array([r["final_loss"] for r in results], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss vs FLOPs (the main scaling law plot)
    ax = axes[0]
    ax.scatter(flops, losses, s=80, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (flops[i], losses[i]), textcoords="offset points", xytext=(8, 5))

    # Fit power law: L = a * C^b
    if len(results) >= 3:
        popt, _ = curve_fit(power_law, flops, losses, p0=[10, -0.05], maxfev=10000)
        x_fit = np.geomspace(flops.min() * 0.5, flops.max() * 10, 100)
        ax.plot(x_fit, power_law(x_fit, *popt), "--", color="gray", alpha=0.7,
                label=f"L = {popt[0]:.2f} * C^({popt[1]:.4f})")
        ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Compute")
    ax.grid(True, alpha=0.3)

    # 2. Loss vs Parameters
    ax = axes[1]
    ax.scatter(params, losses, s=80, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], losses[i]), textcoords="offset points", xytext=(8, 5))

    if len(results) >= 3:
        popt_n, _ = curve_fit(power_law, params, losses, p0=[10, -0.05], maxfev=10000)
        x_fit = np.geomspace(params.min() * 0.5, params.max() * 10, 100)
        ax.plot(x_fit, power_law(x_fit, *popt_n), "--", color="gray", alpha=0.7,
                label=f"L = {popt_n[0]:.2f} * N^({popt_n[1]:.4f})")
        ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Parameters")
    ax.grid(True, alpha=0.3)

    # 3. Loss vs Tokens
    ax = axes[2]
    ax.scatter(tokens, losses, s=80, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(name, (tokens[i], losses[i]), textcoords="offset points", xytext=(8, 5))

    if len(results) >= 3:
        popt_d, _ = curve_fit(power_law, tokens, losses, p0=[10, -0.05], maxfev=10000)
        x_fit = np.geomspace(tokens.min() * 0.5, tokens.max() * 10, 100)
        ax.plot(x_fit, power_law(x_fit, *popt_d), "--", color="gray", alpha=0.7,
                label=f"L = {popt_d[0]:.2f} * D^({popt_d[1]:.4f})")
        ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Tokens")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scaling_laws.png", dpi=150, bbox_inches="tight")
    print("Saved scaling_laws.png")

    # Print fitted exponents
    if len(results) >= 3:
        print(f"Fitted exponents:")
        print(f"Loss vs Compute: L = {popt[0]:.4f} * C^({popt[1]:.6f})")
        print(f"Loss vs Params:  L = {popt_n[0]:.4f} * N^({popt_n[1]:.6f})")
        print(f"Loss vs Tokens:  L = {popt_d[0]:.4f} * D^({popt_d[1]:.6f})")

        # Extrapolate to 1B params
        predicted_1b = power_law(1e9, *popt_n)
        print(f"Predicted loss at 1B params: {predicted_1b:.4f}")

    plt.show()


if __name__ == "__main__":
    main()
