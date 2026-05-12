#!/usr/bin/env python3.12
"""
Draw multiple logistic functions and their summed function.

Function form:

    f(x) = L / (1 + e^(-k * (x - x0)))

Edit the PARAMETERS section below to control each function's L and x0.
"""

from dataclasses import dataclass
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class LogisticFunction:
    L: float
    x0: float
    label: str | None = None


# =========================
# PARAMETERS
# =========================

# Shared steepness. Larger k means the transition is sharper.
K = 1.0

# Add, remove, or edit functions here.
# You only need to change L and x0 for each curve.
FUNCTIONS = [
    LogisticFunction(L=1.0, x0=0.0),
    LogisticFunction(L=2.0, x0=2.0),
    LogisticFunction(L=1.5, x0=4.0),
    LogisticFunction(L=3.0, x0=6.0),
    LogisticFunction(L=2.0, x0=8.0),
]

# X-axis range and sampling density.
X_MIN = -5.0
X_MAX = 20.0
NUM_POINTS = 2000

# Output image path. Set to None if you only want the pop-up window.
OUTPUT_FILE = "sigmoid_sum.png"

# Whether to open a plot window when an interactive backend is available.
SHOW_PLOT = True


def logistic(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    return L / (1 + np.exp(-k * (x - x0)))


def main() -> None:
    x = np.linspace(X_MIN, X_MAX, NUM_POINTS)
    y_sum = np.zeros_like(x)

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, func in enumerate(FUNCTIONS, start=1):
        y = logistic(x, func.L, K, func.x0)
        y_sum += y

        label = func.label or f"f{index}: L={func.L:g}, x0={func.x0:g}"
        ax.plot(x, y, linewidth=1.8, alpha=0.85, label=label)

    ax.plot(x, y_sum, color="black", linewidth=3.0, label="sum")

    ax.set_title(f"Logistic Functions and Sum (k={K:g})")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if OUTPUT_FILE:
        fig.savefig(OUTPUT_FILE, dpi=200)
        print(f"Saved plot to {OUTPUT_FILE}")

    if SHOW_PLOT and "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
