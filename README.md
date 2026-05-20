# AlGe Binary Phase Diagram via UMA-S-1.2 + LAMMPS

Computes the Al–Ge binary phase diagram (liquidus, solidus, eutectic point) using
Meta's [UMA-S-1.2](https://github.com/facebookresearch/fairchem) universal machine-learning
interatomic potential as the force engine inside LAMMPS.

Experimental reference: eutectic at **~28 at% Ge, 697 K (424 °C)**.

---

## Method overview

The pyiron workshop tutorial for binary phase diagrams
([AlLi with EAM](https://workshop.pyiron.org/potentials-workshop-2022/phase_diagram/tutorial_2.html))
uses full thermodynamic integration (Calphy / Einstein-crystal / Uhlenbeck-Ford).
That approach requires multiple LAMMPS `run` commands per job, which conflicts with a
known limitation of the current fairchem-lammps interface (issue #1958).

This workflow uses a simpler **NPT heating-curve** approach instead:

```
For each (composition x_Ge, temperature T):
  1. Start from an ordered solid (FCC-Al or diamond-Ge with random substitutions)
  2. Run a short NPT MD simulation at temperature T
  3. Record mean potential energy PE(T) and volume V(T)

For each composition:
  4. Find T_melt = peak of d(PE)/dT  ← melting onset
  5. Combine Al-rich and Ge-rich liquidus branches
  6. Eutectic = minimum of the combined liquidus curve
```

**Accuracy**: ±20–50 K (vs ±5–10 K for thermodynamic integration).
Sufficient to locate the eutectic and trace liquidus lines.

### Why not use Calphy / thermodynamic integration?

| Constraint | Impact |
|---|---|
| Single `run` command per LAMMPS script (bug #1958) | Breaks Calphy's multi-run lambda-switching workflow |
| UMA `minimize` returns zero energy (bug #1958) | Rules out geometry optimization steps |
| CPU-only inference | Makes long TI lambda-sweeps impractical |

Each heating run is a separate Python-orchestrated LAMMPS job with one `run` command,
which works around both bugs.

---

## Repository layout

```
.
├── 00_test_uma_lammps.py        # Smoke test — run this first
├── 01_build_structures.py       # Generate FCC-Al / diamond-Ge supercells
├── 02_run_heating.py            # Orchestrate UMA+LAMMPS NPT runs across (x, T) grid
├── 03_analyze_melting.py        # Detect T_melt per composition, write CSV
├── 04_plot_phase_diagram.py     # Plot T-x diagram + eutectic estimate
│
├── lammps_fc_config.yaml        # UMA-S-1.2 config (device, model name)
├── lammps_input_template.lammps # Annotated NPT LAMMPS input template (reference)
│
├── environment.yml              # Conda environment (Linux / macOS / Windows)
├── requirements.txt             # Pip-only dependency list
├── setup_env.bat                # Windows one-shot setup script (uses uv)
├── activate.bat                 # Windows activation script (sets LAMMPS PATH)
│
├── structures/                  # LAMMPS data files (created by 01_)
├── runs/heating/                # Per-run LAMMPS inputs and thermo logs (created by 02_)
└── results/                     # JSON results per (x_Ge, T) point (created by 02_)
```

---

## Quick start

### 1. Create the environment

**Linux / macOS (conda):**
```bash
conda env create -f environment.yml
conda activate uma-alge-phase-diagrams
```

**Windows (uv, one-time setup):**
```bat
setup_env.bat          # creates .venv with Python 3.11 + all deps
activate.bat           # use this instead of .venv\Scripts\activate.bat
```

### 2. Validate the environment
```bash
python 00_test_uma_lammps.py
```

Expected output:
```
Wrote 8-atom FCC Al cell to al_test.lammps
Loading UMA-S-1.2 predictor...
Predictor loaded.
Running 100-step NVE test...

PE/atom = -3.73 eV/atom  (reference FCC Al ~-3.36 eV/atom)

SMOKE TEST PASSED — UMA-S-1.2 + LAMMPS is working.
```

If `PotEng` is zero, this indicates fairchem-lammps bug #1958 — check your
`fairchem-lammps` version (`>= 0.4.0` required).

### 3. Build AlGe structures
```bash
python 01_build_structures.py
```
Writes 13 LAMMPS data files to `structures/` covering
`x_Ge = 0.00, 0.05, 0.10, …, 0.80, 1.00`.

- `x_Ge ≤ 0.50` → 4×4×4 FCC-Al supercell (256 atoms) with random Ge substitutions
- `x_Ge > 0.50` → 3×3×3 diamond-Ge supercell (216 atoms) with random Al substitutions

### 4. Run the heating scan

**Quick test first** (2 compositions, 3 temperatures, 200 steps — ~30 min on CPU):
```bash
python 02_run_heating.py --test
```

**Full production run** (13 compositions × 12 temperatures × 5000 steps):
```bash
python 02_run_heating.py
```

**Parallel execution** — run N jobs simultaneously across CPU cores:
```bash
python 02_run_heating.py --workers 4
```

Each worker is an independent subprocess that loads its own model copy. Allow
~4 GB RAM per worker. On a 16 GB machine, `--workers 3` or `--workers 4` is
safe. Jobs are distributed round-robin across workers so each process handles
a mix of compositions rather than a contiguous block.

> **CPU performance note**: UMA inference on CPU runs at roughly 2–3 steps/sec
> for 256-atom cells (~12 s/step). The full production grid is intended for GPU
> execution. On CPU, validate with `--test` first, then consider `--workers N`
> to exploit multiple cores while waiting for GPU access.

Results are cached in `results/alge_x{x}_T{T}.json` — interrupted runs resume
automatically (already-cached jobs are skipped by all workers on restart).

### 5. Detect melting points
```bash
python 03_analyze_melting.py
```
Outputs:
- `melting_points.csv` — `x_ge, T_melt_K, pe_jump_eV, vol_jump_A3`
- `melting_curves.png` — PE(T) curves with detected T_melt marked per composition

### 6. Plot the phase diagram
```bash
python 04_plot_phase_diagram.py
```
Outputs `phase_diagram_AlGe.png` with:
- UMA-computed liquidus branches (Al-rich and Ge-rich)
- Eutectic estimate (minimum of the liquidus)
- Experimental eutectic (28 at% Ge, 697 K) for comparison

---

## AlGe system facts

| Property | Value |
|---|---|
| Al crystal structure | FCC, a = 4.05 Å |
| Ge crystal structure | Diamond cubic, a = 5.66 Å |
| Phase diagram type | Simple eutectic (no stable intermetallics) |
| Experimental eutectic composition | ~28 at% Ge |
| Experimental eutectic temperature | 697 K (424 °C) |
| Solid phases | FCC-Al(Ge) + diamond-Ge(Al) |

---

## UMA + LAMMPS integration details

UMA does **not** use a traditional `pair_style` in LAMMPS. Instead, the
`fairchem-lammps` package registers a Python callback via LAMMPS's external fix
mechanism:

```
fix ext_fc all external pf/callback 1 1
```

The `run_lammps_with_fairchem()` Python function:
1. Separates `run N` commands from the rest of the input script
2. Executes all non-run commands (structure setup, thermostat, thermo output)
3. Injects the external callback fix
4. Runs the MD

**Do not add `pair_style` or `fix external` to your LAMMPS input files** — the
wrapper handles this. Adding them will cause incorrect double-counting of forces.

### Windows DLL conflict workaround

LAMMPS ships its own `libssl-3-x64.dll` in its `bin/` directory. On Windows,
loading the LAMMPS shared library before Python's SSL module causes:
```
ImportError: DLL load failed while importing _ssl
```

**Fix**: import `ssl` at the top of any script that uses LAMMPS, before the first
`lammps()` instantiation. All scripts in this repo already do this.

---

## Known limitations and open issues

| Issue | Status | Workaround |
|---|---|---|
| Single `run` command per script (#1958) | Open | One Python-orchestrated subprocess per (x, T) |
| `minimize` returns zero energy (#1958) | Open | Use NPT MD only, never `minimize` |
| Multi-node inference (#1949) | Open | Single-node only; not relevant for CPU |
| Heating-curve ±20–50 K accuracy | By design | Upgrade to TI when bugs are fixed |

---

## Upgrading to thermodynamic integration

Once fairchem-lammps bug #1958 is resolved, this workflow can be upgraded to
full thermodynamic integration (same method as the pyiron EAM tutorial) by:

1. Using [Calphy](https://calphy.org/) with its ASE kernel (`kernel="ase"`) and
   a `FAIRChemCalculator` wrapping UMA-S-1.2
2. Computing Einstein-crystal free energies for solids and Uhlenbeck-Ford free
   energies for the liquid phase
3. Applying common-tangent construction instead of the heating-curve peak method

This would reduce the eutectic temperature uncertainty from ±50 K to ±10 K.
