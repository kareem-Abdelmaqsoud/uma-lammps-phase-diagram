"""
Heating scan: for each (x_Ge, T) point, run a short NPT MD simulation with
UMA-S-1.2 + LAMMPS and record mean potential energy and volume.

The UMA predictor is loaded ONCE per worker process and reused across that
worker's batch of jobs. With --workers N, the job grid is split across N
independent processes running in parallel.

Results are saved to results/alge_x{x:.2f}_T{T:04d}.json.
Completed jobs are skipped automatically on restart.

Run:
    python 02_run_heating.py [--test] [--workers N] [--device cpu|cuda]

  --test       : small run (2 compositions, 3 temperatures, 200 steps)
  --workers N  : run N jobs in parallel (default 1); each loads its own
                 model copy — ensure you have N × ~4 GB RAM available
  --device     : cpu (default) or cuda
"""

# ssl must be imported before lammps instantiation to prevent DLL conflict on Windows:
# LAMMPS loads its own libssl-3-x64.dll which shadows Python's version if loaded first.
import ssl

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
STRUCTURES_DIR = PROJECT_ROOT / "structures"
RUNS_DIR = PROJECT_ROOT / "runs" / "heating"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

X_GE_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.35, 0.45,
              0.55, 0.65, 0.80, 1.00]

T_GRID = [400, 500, 600, 650, 700, 750, 800, 850, 900, 1000, 1100, 1200]

NSTEPS_PRODUCTION = 5000
NSTEPS_TEST = 200


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

    print(f"[worker pid={_pid()}] Loading UMA-S-1.2 predictor...")
    model_path = pretrained_checkpoint_path_from_name("uma-s-1p1")
    predictor = load_predict_unit(model_path, device=device, inference_settings=settings)
    print(f"[worker pid={_pid()}] Predictor ready.")
    return predictor


def _pid() -> int:
    import os
    return os.getpid()


def structure_label(x_ge: float) -> str:
    return "fcc" if x_ge <= 0.50 else "diamond"


def make_lammps_input(data_file: Path, log_file: Path, temp: float,
                      nsteps: int, seed: int) -> str:
    return (
        f"units           metal\n"
        f"atom_style      atomic\n"
        f"boundary        p p p\n"
        f"\n"
        f"read_data       {data_file.as_posix()}\n"
        f"\n"
        f"mass 1 26.982\n"
        f"mass 2 72.630\n"
        f"\n"
        f"velocity all create {temp:.1f} {seed} mom yes rot yes dist gaussian\n"
        f"\n"
        f"fix 1 all npt temp {temp:.1f} {temp:.1f} 0.1 iso 0.0 0.0 1.0\n"
        f"\n"
        f"thermo 50\n"
        f"thermo_style custom step temp pe ke etotal press vol\n"
        f"\n"
        f"log {log_file.as_posix()}\n"
        f"\n"
        f"timestep 0.002\n"
        f"run {nsteps}\n"
    )


def parse_thermo_log(log_file: Path) -> list[dict]:
    if not log_file.exists():
        return []
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
                header = None
    return rows


def extract_averages(rows: list[dict], discard_fraction: float = 0.5) -> dict:
    if not rows:
        return {}
    n_discard = int(len(rows) * discard_fraction)
    production = rows[n_discard:]
    keys = production[0].keys()
    return {k: float(np.mean([r[k] for r in production])) for k in keys}


def run_one(predictor, x_ge: float, temp: float, nsteps: int, seed: int) -> dict | None:
    from fairchem.lammps.lammps_fc import run_lammps_with_fairchem

    label = structure_label(x_ge)
    data_file = STRUCTURES_DIR / f"alge_x{x_ge:.2f}_{label}.lammps"
    if not data_file.exists():
        print(f"  SKIP: structure file not found: {data_file.name}")
        return None

    run_tag = f"x{x_ge:.2f}_T{temp:04.0f}"
    run_dir = RUNS_DIR / run_tag
    run_dir.mkdir(exist_ok=True)

    log_file = run_dir / "thermo.log"
    input_file = run_dir / "input.lammps"
    input_file.write_text(make_lammps_input(data_file, log_file, temp, nsteps, seed))

    try:
        lmp = run_lammps_with_fairchem(predictor, str(input_file), task_name="omat")
        del lmp._predictor
    except Exception as e:
        print(f"  ERROR {run_tag}: {e}")
        return None

    rows = parse_thermo_log(log_file)
    if not rows:
        print(f"  WARNING: no thermo data for {run_tag}")
        return None

    avgs = extract_averages(rows)
    if abs(avgs.get("PotEng", 0.0)) < 1e-6:
        print(f"  WARNING: PotEng~0 for {run_tag} — possible bug #1958")
    return avgs


def save_result(x_ge: float, temp: float, avgs: dict):
    out = RESULTS_DIR / f"alge_x{x_ge:.2f}_T{temp:04.0f}.json"
    payload = {"x_ge": x_ge, "T_K": temp, **avgs}
    out.write_text(json.dumps(payload, indent=2))


# ── worker entry point (runs in a subprocess) ──────────────────────────────
def worker_batch(jobs: list[tuple[float, float]], device: str, nsteps: int) -> int:
    """Load predictor once, then run every (x_ge, T) job in this batch.

    Called by ProcessPoolExecutor in a fresh subprocess. Returns count of
    completed jobs.
    """
    # ssl pre-import needed in every subprocess on Windows
    import ssl  # noqa: F401

    predictor = load_uma_predictor(device=device)
    completed = 0
    for x_ge, temp in jobs:
        result_file = RESULTS_DIR / f"alge_x{x_ge:.2f}_T{temp:04.0f}.json"
        if result_file.exists():
            print(f"  [pid={_pid()}] x={x_ge:.2f} T={temp:4d}K — cached, skip")
            continue
        print(f"  [pid={_pid()}] x={x_ge:.2f} T={temp:4d}K ...", flush=True)
        seed = int(x_ge * 1000 + temp)
        avgs = run_one(predictor, x_ge, temp, nsteps, seed)
        if avgs:
            save_result(x_ge, temp, avgs)
            pe = avgs.get("PotEng", float("nan"))
            print(f"  [pid={_pid()}] x={x_ge:.2f} T={temp:4d}K  PE={pe:.3f} eV  done")
            completed += 1
        else:
            print(f"  [pid={_pid()}] x={x_ge:.2f} T={temp:4d}K  FAILED")
    return completed


# ── sequential runner (workers=1) ──────────────────────────────────────────
def run_sequential(x_grid, t_grid, nsteps, device):
    predictor = load_uma_predictor(device=device)
    total = len(x_grid) * len(t_grid)
    done = 0
    for x_ge in x_grid:
        for temp in t_grid:
            done += 1
            result_file = RESULTS_DIR / f"alge_x{x_ge:.2f}_T{temp:04.0f}.json"
            tag = f"x={x_ge:.2f}  T={temp:4d} K"
            if result_file.exists():
                print(f"  [{done:3d}/{total}] {tag}  — cached, skip")
                continue
            print(f"  [{done:3d}/{total}] {tag} ...", end=" ", flush=True)
            seed = int(x_ge * 1000 + temp)
            avgs = run_one(predictor, x_ge, temp, nsteps, seed)
            if avgs:
                save_result(x_ge, temp, avgs)
                print(f"PE={avgs.get('PotEng', float('nan')):.3f} eV  done")
            else:
                print("FAILED")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AlGe heating scan using UMA-S-1.2 + LAMMPS"
    )
    parser.add_argument("--test", action="store_true",
                        help="2 compositions x 3 temperatures x 200 steps")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default 1). "
                             "Each loads its own model — allow ~4 GB RAM per worker.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    if args.test:
        x_grid = [0.00, 0.28]
        t_grid = [700, 900, 1100]
        nsteps = NSTEPS_TEST
        print("TEST MODE: 2 compositions x 3 temperatures x 200 steps")
    else:
        x_grid = X_GE_GRID
        t_grid = T_GRID
        nsteps = NSTEPS_PRODUCTION
        print(f"PRODUCTION: {len(x_grid)} comps x {len(t_grid)} temps x {nsteps} steps")

    workers = max(1, args.workers)
    total_jobs = len(x_grid) * len(t_grid)
    print(f"Workers: {workers}  |  Total jobs: {total_jobs}  |  Device: {args.device}\n")

    if workers == 1:
        run_sequential(x_grid, t_grid, nsteps, args.device)
    else:
        # Build flat list of pending jobs (skip already-cached results)
        all_jobs = [
            (x, t)
            for x in x_grid
            for t in t_grid
            if not (RESULTS_DIR / f"alge_x{x:.2f}_T{t:04.0f}.json").exists()
        ]
        cached = total_jobs - len(all_jobs)
        print(f"Cached: {cached}  |  To run: {len(all_jobs)}")

        if not all_jobs:
            print("All jobs already cached.")
        else:
            # Round-robin partition into worker batches so each worker gets
            # a mix of compositions rather than a contiguous block.
            batches = [all_jobs[i::workers] for i in range(workers)]
            batches = [b for b in batches if b]  # drop empty batches

            print(f"Dispatching {len(batches)} batches across {workers} workers...\n")
            total_completed = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(worker_batch, batch, args.device, nsteps): i
                    for i, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    worker_id = futures[future]
                    try:
                        n = future.result()
                        total_completed += n
                        print(f"\n[worker {worker_id}] finished — {n} jobs completed")
                    except Exception as exc:
                        print(f"\n[worker {worker_id}] raised: {exc}")

            print(f"\nAll workers done. {total_completed} new results saved.")

    print(f"\nResults in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
