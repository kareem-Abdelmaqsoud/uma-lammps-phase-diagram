"""
Build and plot the Al-Ge T-x phase diagram from the melting points in
melting_points.csv (produced by 03_analyze_melting.py).

The phase diagram shows:
  - Liquidus on the Al-rich side  (FCC-Al + liquid boundary)
  - Liquidus on the Ge-rich side  (diamond-Ge + liquid boundary)
  - Estimated eutectic point      (minimum of the combined liquidus)
  - Experimental eutectic         (28 at% Ge, 697 K) for reference

Run:
    python 04_plot_phase_diagram.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).parent
CSV_PATH = PROJECT_ROOT / "melting_points.csv"

# Experimental eutectic reference
EXP_EUTECTIC_X = 0.28   # at% Ge (mole fraction)
EXP_EUTECTIC_T = 697.0  # K


def load_melting_points(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    x, T = [], []
    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            x.append(float(parts[0]))
            T.append(float(parts[1]))
    return np.array(x), np.array(T)


def fit_liquidus_branch(x: np.ndarray, T: np.ndarray,
                        x_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2nd-degree polynomial to one liquidus branch and return (x_fine, T_fit)."""
    mask = x <= x_max
    if mask.sum() < 2:
        return np.array([]), np.array([])
    coeffs = np.polyfit(x[mask], T[mask], 2)
    x_fine = np.linspace(x[mask].min(), x_max, 200)
    return x_fine, np.polyval(coeffs, x_fine)


def find_eutectic(x_al_branch: np.ndarray, T_al_branch: np.ndarray,
                  x_ge_branch: np.ndarray, T_ge_branch: np.ndarray
                  ) -> tuple[float, float]:
    """Approximate eutectic as the intersection of the two liquidus polynomial fits."""
    x_full = np.concatenate([x_al_branch, x_ge_branch])
    T_full = np.concatenate([T_al_branch, T_ge_branch])
    idx = np.argmin(T_full)
    return float(x_full[idx]), float(T_full[idx])


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH.name} not found — run 03_analyze_melting.py first.")
        return

    x, T = load_melting_points(CSV_PATH)
    if len(x) == 0:
        print("No melting points loaded.")
        return

    print(f"Loaded {len(x)} melting points.")

    # Split into Al-rich and Ge-rich branches at the eutectic composition region
    # Use x<=0.35 for Al branch, x>=0.20 for Ge branch (overlapping window near eutectic)
    x_al_fit, T_al_fit = fit_liquidus_branch(x, T, x_max=0.50)
    x_ge_fit, T_ge_fit = fit_liquidus_branch(1.0 - x, T, x_max=0.50)
    # Mirror Ge branch back to original x-axis
    if len(x_ge_fit) > 0:
        x_ge_fit = 1.0 - x_ge_fit

    # Eutectic estimate
    if len(x_al_fit) > 0 and len(x_ge_fit) > 0:
        x_eut, T_eut = find_eutectic(x_al_fit, T_al_fit, x_ge_fit, T_ge_fit)
    else:
        x_eut, T_eut = float("nan"), float("nan")

    # ---- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Raw melting points
    ax.scatter(x * 100, T, color="steelblue", zorder=5, s=50, label="UMA-S-1.2 T_melt")

    # Liquidus fits
    if len(x_al_fit) > 0:
        ax.plot(x_al_fit * 100, T_al_fit, "b-", lw=2,
                label="Al-rich liquidus (fit)")
    if len(x_ge_fit) > 0:
        ax.plot(x_ge_fit * 100, T_ge_fit, "g-", lw=2,
                label="Ge-rich liquidus (fit)")

    # UMA eutectic estimate
    if not np.isnan(x_eut):
        ax.scatter([x_eut * 100], [T_eut], marker="*", s=200, color="red", zorder=6,
                   label=f"UMA eutectic ≈ ({x_eut*100:.0f} at% Ge, {T_eut:.0f} K)")

    # Experimental eutectic
    ax.scatter([EXP_EUTECTIC_X * 100], [EXP_EUTECTIC_T], marker="^", s=120,
               color="orange", zorder=6,
               label=f"Exp. eutectic (28 at%, 697 K)")

    # Region labels
    ax.text(5, 450, "FCC-Al\n+ Liq", ha="center", fontsize=9, color="navy")
    ax.text(90, 450, "diamond-Ge\n+ Liq", ha="center", fontsize=9, color="darkgreen")
    ax.text(50, 780, "Liquid", ha="center", fontsize=11, color="gray")

    ax.set_xlabel("Composition (at% Ge)", fontsize=12)
    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.set_title("Al–Ge Binary Phase Diagram (UMA-S-1.2 heating curve method)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(300, 1350)
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_png = PROJECT_ROOT / "phase_diagram_AlGe.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved phase diagram to {out_png.name}")

    if not np.isnan(x_eut):
        print(f"\nUMA eutectic estimate : {x_eut*100:.1f} at% Ge,  {T_eut:.0f} K")
        print(f"Experimental eutectic : {EXP_EUTECTIC_X*100:.0f} at% Ge,  {EXP_EUTECTIC_T:.0f} K")
        dx = abs(x_eut - EXP_EUTECTIC_X) * 100
        dT = abs(T_eut - EXP_EUTECTIC_T)
        print(f"Deviation             : Δx={dx:.1f} at%,  ΔT={dT:.0f} K")

    plt.show()


if __name__ == "__main__":
    main()
