"""
Detect the melting temperature for each composition from the NPT heating scan
results produced by 02_run_heating.py.

Method: the melting transition appears as a sharp jump in potential energy
(and volume) vs temperature. We locate it via the maximum of d(PE)/dT.

Outputs:
    melting_points.csv   — columns: x_ge, T_melt_K, pe_jump, vol_jump
    melting_curves.png   — PE(T) curves with detected T_melt marked

Run:
    python 03_analyze_melting.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"

X_GE_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.35, 0.45,
              0.55, 0.65, 0.80, 1.00]


def load_composition_data(x_ge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_array, pe_array, vol_array) for a given composition."""
    files = sorted(RESULTS_DIR.glob(f"alge_x{x_ge:.2f}_T*.json"))
    if not files:
        return np.array([]), np.array([]), np.array([])

    T_vals, pe_vals, vol_vals = [], [], []
    for f in files:
        d = json.loads(f.read_text())
        T_vals.append(d["T_K"])
        pe_vals.append(d.get("PotEng", float("nan")))
        vol_vals.append(d.get("Volume", float("nan")))

    order = np.argsort(T_vals)
    return (np.array(T_vals)[order],
            np.array(pe_vals)[order],
            np.array(vol_vals)[order])


def detect_melting_temperature(T: np.ndarray, pe: np.ndarray) -> float | None:
    """Return T where d(PE)/dT is maximum (largest discontinuity = melting)."""
    if len(T) < 3:
        return None
    valid = ~np.isnan(pe)
    if valid.sum() < 3:
        return None
    T_v, pe_v = T[valid], pe[valid]

    dpe_dT = np.gradient(pe_v, T_v)
    idx_max = np.argmax(dpe_dT)

    # Interpolate between the two surrounding temperature points for a finer estimate
    if 0 < idx_max < len(T_v) - 1:
        T_lo, T_hi = T_v[idx_max - 1], T_v[idx_max + 1]
        T_melt = 0.5 * (T_lo + T_hi)
    else:
        T_melt = T_v[idx_max]

    return float(T_melt)


def pe_and_vol_jumps(T: np.ndarray, pe: np.ndarray, vol: np.ndarray,
                     T_melt: float) -> tuple[float, float]:
    """Estimate the PE and volume jump across the melting transition."""
    below = T < T_melt
    above = T > T_melt
    if below.sum() == 0 or above.sum() == 0:
        return float("nan"), float("nan")
    pe_jump = np.nanmean(pe[above][:2]) - np.nanmean(pe[below][-2:])
    vol_jump = np.nanmean(vol[above][:2]) - np.nanmean(vol[below][-2:])
    return float(pe_jump), float(vol_jump)


def main():
    results = []
    fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharey=False)
    axes = axes.flatten()

    for ax_idx, x_ge in enumerate(X_GE_GRID):
        T, pe, vol = load_composition_data(x_ge)
        ax = axes[ax_idx] if ax_idx < len(axes) else None

        if len(T) == 0:
            print(f"x_Ge={x_ge:.2f}  — no data found, skipping")
            if ax:
                ax.set_title(f"x_Ge={x_ge:.2f}\nno data")
                ax.axis("off")
            continue

        T_melt = detect_melting_temperature(T, pe)
        pe_jump, vol_jump = (pe_and_vol_jumps(T, pe, vol, T_melt)
                             if T_melt is not None else (float("nan"), float("nan")))

        if T_melt is not None:
            print(f"x_Ge={x_ge:.2f}  T_melt={T_melt:6.0f} K  "
                  f"ΔPE={pe_jump:+.3f} eV  ΔVol={vol_jump:+.1f} Å³")
            results.append({"x_ge": x_ge, "T_melt_K": T_melt,
                             "pe_jump_eV": pe_jump, "vol_jump_A3": vol_jump})
        else:
            print(f"x_Ge={x_ge:.2f}  — not enough data to detect T_melt")

        if ax:
            valid = ~np.isnan(pe)
            ax.plot(T[valid], pe[valid], "o-", ms=4, color="steelblue")
            if T_melt is not None:
                ax.axvline(T_melt, color="red", ls="--", lw=1.2,
                           label=f"T_melt={T_melt:.0f} K")
                ax.legend(fontsize=7)
            ax.set_title(f"x_Ge={x_ge:.2f}", fontsize=9)
            ax.set_xlabel("T (K)", fontsize=8)
            ax.set_ylabel("PE (eV)", fontsize=8)

    # Hide unused axes
    for i in range(len(X_GE_GRID), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("PE vs T heating curves — AlGe system (UMA-S-1.2)", fontsize=12)
    fig.tight_layout()
    out_png = PROJECT_ROOT / "melting_curves.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved heating curves to {out_png.name}")

    if not results:
        print("No melting points detected — run 02_run_heating.py first.")
        return

    # Write CSV
    csv_path = PROJECT_ROOT / "melting_points.csv"
    with open(csv_path, "w") as f:
        f.write("x_ge,T_melt_K,pe_jump_eV,vol_jump_A3\n")
        for r in sorted(results, key=lambda d: d["x_ge"]):
            f.write(f"{r['x_ge']:.4f},{r['T_melt_K']:.1f},"
                    f"{r['pe_jump_eV']:.4f},{r['vol_jump_A3']:.2f}\n")
    print(f"Saved melting points to {csv_path.name}")


if __name__ == "__main__":
    main()
