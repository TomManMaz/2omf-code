# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements exact and heuristic algorithms for the **2-Optimality Motif Finding (2-OMF)** problem, accompanying the paper *"Advancing the 2-Optimality Motif Finding problem: new benchmarks and efficient algorithms"*. Given n integer sequences of length m over an alphabet of size k, the objective is to find a "median" string that minimizes the sum of squared Hamming distances to all input sequences.

## Setup and Running

```bash
pip install -r requirements.txt
bash 1-preprocess.sh  # extract benchmark instances from instances.tar.gz
```

Run an algorithm:
```bash
python main.py -a sa -i instances/uniform_480_50_2_2.dat --timeout 20
python main.py -a hg2 -i instances/uniform_480_50_2_2.dat --timeout 20
python main.py -a gurobi -i instances/uniform_480_50_2_2.dat --timeout 20  # requires gurobipy
```

Available algorithms (`-a`): `sa`, `simulated_annealing`, `hg2`, `gurobi`.

There is no test suite in the repository. No linter or formatter is configured.

## Architecture

**Entry point:** `main.py` parses CLI args and calls `minimize()`.

**Dispatch layer:** `algorithms/minimize.py` — the `minimize()` function routes to the appropriate algorithm class based on the `method` string. Algorithm-specific options are passed via the `options` dict.

**Algorithms** (each in `algorithms/`):
- `simulated_annealing.py` — `SimulatedAnnealing` class. Time-based cooling schedule (results depend on wall-clock time, so not perfectly reproducible across hardware). Uses delta evaluation for O(n) neighborhood moves.
- `hg2.py` — `HG2` class. Hybrid Genetic Algorithm using pure NumPy arrays (no `Solution` objects in hot paths). Population stored as 2D int32 array.
- `gurobi.py` — `GurobiWrapper` class. Exact ILP solver; optional dependency (`gurobipy`).

**Data model** (in `data/`):
- `Instance` — holds n sequences as an `np.ndarray` of shape `(n, m)`. Loaded from `.dat` files via `Instance.from_file()`. Also provides lower bound computations (`static_bound`, `mode_bound`) and instance generation methods.
- `Solution` — wraps a candidate sequence + its instance. `evaluate()` computes sum of squared Hamming distances. `calculate_objective_delta()` provides O(n) incremental evaluation for single-character changes (used by SA).
- `Result` (extends `_NiceResult` from `_lib/_util.py`) — optimization result container with `objective_value`, `x`, `duration`, `n_iterations`, `gap`, etc. Results are appended to CSV files via `to_file()`.

**Config:** `config/algorithm_params.py` — IRACE-tuned default parameters for SA and HG2 as frozen dataclasses (`SimulatedAnnealingConfig`, `HG2Config`).

**Output:** Results are written to `~/2omf/experiments/summary/{instance}_{algorithm}_{seed}.csv`.

## Key Conventions

- All algorithms return a `Result` object from their minimize method.
- Instance files use the format: first line is `n m k`, followed by n rows of m space-separated integers.
- The objective function is **sum of squared Hamming distances** (not plain Hamming).
- `Solution._distances` caches per-sequence Hamming distances and must be kept in sync when modifying the solution sequence.
- SA and HG2 default parameters come from `config/algorithm_params.py`; CLI args override them via `options` dict passthrough.
