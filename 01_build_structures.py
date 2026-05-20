"""
Build FCC-Al (Al-rich) and diamond-Ge (Ge-rich) supercells with random substitutions
at a grid of Al-Ge compositions. Writes LAMMPS data files to structures/.

Atom type mapping (consistent across all files):
    type 1 = Al  (mass 26.982)
    type 2 = Ge  (mass 72.630)

Run:
    python 01_build_structures.py
"""

import random
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write as ase_write

PROJECT_ROOT = Path(__file__).parent
STRUCTURES_DIR = PROJECT_ROOT / "structures"
STRUCTURES_DIR.mkdir(exist_ok=True)

# Ge mole fractions to sample.
# Near the eutectic (~0.28) we sample more densely.
X_GE_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.35, 0.45,
              0.55, 0.65, 0.80, 1.00]

# Reproducible random substitutions
RANDOM_SEED = 42


def build_fcc_al_with_ge(x_ge: float, supercell: tuple = (4, 4, 4)) -> object:
    """FCC Al supercell with x_ge fraction of sites replaced by Ge."""
    atoms = bulk("Al", "fcc", a=4.050) * supercell
    n_total = len(atoms)
    n_ge = round(n_total * x_ge)

    rng = random.Random(RANDOM_SEED)
    ge_indices = rng.sample(range(n_total), n_ge)
    symbols = atoms.get_chemical_symbols()
    for i in ge_indices:
        symbols[i] = "Ge"
    atoms.set_chemical_symbols(symbols)
    return atoms


def build_diamond_ge_with_al(x_ge: float, supercell: tuple = (3, 3, 3)) -> object:
    """Diamond-cubic Ge supercell (conventional cell × supercell) with (1-x_ge)
    fraction of sites replaced by Al."""
    # cubic=True gives the 8-atom conventional cell; 3×3×3 → 216 atoms
    atoms = bulk("Ge", "diamond", a=5.658, cubic=True) * supercell
    n_total = len(atoms)
    n_al = round(n_total * (1.0 - x_ge))

    rng = random.Random(RANDOM_SEED)
    al_indices = rng.sample(range(n_total), n_al)
    symbols = atoms.get_chemical_symbols()
    for i in al_indices:
        symbols[i] = "Al"
    atoms.set_chemical_symbols(symbols)
    return atoms


def write_structure(atoms, x_ge: float, label: str) -> Path:
    out = STRUCTURES_DIR / f"alge_x{x_ge:.2f}_{label}.lammps"
    ase_write(str(out), atoms, format="lammps-data", specorder=["Al", "Ge"])
    return out


def main():
    print(f"Writing LAMMPS data files to {STRUCTURES_DIR}/\n")
    print(f"{'x_Ge':>6}  {'structure':>10}  {'n_atoms':>7}  {'n_Al':>6}  {'n_Ge':>6}  file")
    print("-" * 75)

    for x_ge in X_GE_GRID:
        if x_ge <= 0.50:
            atoms = build_fcc_al_with_ge(x_ge)
            label = "fcc"
        else:
            atoms = build_diamond_ge_with_al(x_ge)
            label = "diamond"

        syms = atoms.get_chemical_symbols()
        n_al = syms.count("Al")
        n_ge = syms.count("Ge")
        path = write_structure(atoms, x_ge, label)

        print(f"{x_ge:6.2f}  {label:>10}  {len(atoms):>7}  {n_al:>6}  {n_ge:>6}  {path.name}")

    print(f"\nDone. {len(X_GE_GRID)} structures written.")


if __name__ == "__main__":
    main()
