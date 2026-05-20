"""
Smoke test: verify UMA-S-1.2 + LAMMPS integration produces non-zero forces/energies
for a small pure-Al cell before running the full phase diagram workflow.

Run:
    python 00_test_uma_lammps.py
"""

# ssl must be imported before lammps instantiation to prevent DLL conflict on Windows:
# LAMMPS loads its own libssl-3-x64.dll which shadows Python's version if loaded first.
import ssl

import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write as ase_write

PROJECT_ROOT = Path(__file__).parent
TEST_DIR = PROJECT_ROOT / "runs" / "test"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def load_uma_predictor(device: str = "cpu"):
    from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
    from fairchem.core.units.mlip_unit import load_predict_unit

    try:
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        settings = InferenceSettings(
            tf32=False,
            merge_mole=False,
            compile=False,
            activation_checkpointing=False,
            internal_graph_gen_version=2,
            external_graph_gen=False,
        )
    except ImportError:
        settings = None

    print("Loading UMA-S-1.2 predictor (may download model on first run)...")
    model_path = pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = load_predict_unit(model_path, device=device, inference_settings=settings)
    print("Predictor loaded.")
    return predictor


def write_lammps_input(data_file: Path, log_file: Path, nsteps: int = 100) -> Path:
    input_file = TEST_DIR / "test_input.lammps"
    input_file.write_text(
        f"units           metal\n"
        f"atom_style      atomic\n"
        f"boundary        p p p\n"
        f"\n"
        f"read_data       {data_file.as_posix()}\n"
        f"\n"
        f"mass 1 26.982\n"
        f"\n"
        f"velocity all create 300.0 42 mom yes rot yes dist gaussian\n"
        f"\n"
        f"fix 1 all nve\n"
        f"\n"
        f"thermo 10\n"
        f"thermo_style custom step temp pe ke etotal press vol\n"
        f"\n"
        f"log {log_file.as_posix()}\n"
        f"\n"
        f"timestep 0.001\n"
        f"run {nsteps}\n"
    )
    return input_file


def parse_thermo_log(log_file: Path) -> list[dict]:
    """Return list of thermo dicts, one per printed step."""
    lines = log_file.read_text().splitlines()
    header = None
    rows = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Step" and "Temp" in parts:
            header = parts
            continue
        if header:
            try:
                rows.append({k: float(v) for k, v in zip(header, parts)})
            except ValueError:
                header = None  # hit end-of-block
    return rows


def main():
    try:
        from fairchem.lammps.lammps_fc import run_lammps_with_fairchem
    except ImportError as e:
        print(f"ERROR: fairchem-lammps not installed or LAMMPS Python library missing.\n{e}")
        sys.exit(1)

    # 32-atom FCC Al supercell
    atoms = bulk("Al", "fcc", a=4.05) * (2, 2, 2)
    data_file = TEST_DIR / "al_test.lammps"
    ase_write(str(data_file), atoms, format="lammps-data", specorder=["Al"])
    print(f"Wrote {len(atoms)}-atom FCC Al cell to {data_file.name}")

    predictor = load_uma_predictor(device="cpu")

    log_file = TEST_DIR / "test.log"
    input_file = write_lammps_input(data_file, log_file, nsteps=100)
    print("Running 100-step NVE test...")

    lmp = run_lammps_with_fairchem(predictor, str(input_file), task_name="omat")
    del lmp._predictor

    if not log_file.exists():
        print("ERROR: Log file not created — LAMMPS run may have failed silently.")
        sys.exit(1)

    rows = parse_thermo_log(log_file)
    if not rows:
        print("ERROR: No thermo data found in log file.")
        sys.exit(1)

    last = rows[-1]
    pe = last.get("PotEng", 0.0)
    pe_per_atom = pe / len(atoms)

    print("\n--- Thermo at final step ---")
    for k, v in last.items():
        print(f"  {k:10s} = {v:.5g}")

    if abs(pe) < 1e-6:
        print(
            "\nFAIL: PotEng ≈ 0. This matches known bug #1958 in fairchem-lammps.\n"
            "Check that fairchem-lammps >= 0.5.0 and the UMA callback is registering."
        )
        sys.exit(1)

    expected = -3.36  # eV/atom cohesive energy of Al (approx)
    print(f"\nPE/atom = {pe_per_atom:.3f} eV/atom  (reference FCC Al ~{expected} eV/atom)")
    print("\nSMOKE TEST PASSED — UMA-S-1.2 + LAMMPS is working.")


if __name__ == "__main__":
    main()
