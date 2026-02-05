# 2-Optimality Motif Finding (2-OMF)

**Maintainer:** Tommaso Mannelli Mazzoli

**Contributors:** Tommaso Mannelli Mazzoli, Fulvio Gesmundo, Pedro Pinacho-Davidson, Felix Winter, Christian Blum

## Description
Implementation of exact and heuristic algorithms for the 2-Optimality Motif Finding (2OMF) problem.

## Installation
```bash
git clone [url repo]
cd 2omf
pip install -r requirements.txt
```

## Requirements
- Python >= 3.9 (tested on 3.13.5)

## Optional: Gurobi Solver

The exact optimization method requires Gurobi (commercial solver).

### Without Gurobi
The following algorithms work without Gurobi:
- `sa` / `simulated_annealing`: Simulated Annealing
- `hg2`: Hybrid Genetic Algorithm

### With Gurobi
1. Obtain a license from https://www.gurobi.com/ (Free academic licenses available)
2. Install: `pip install gurobipy`
3. Use: `python main.py -a gurobi -i instances/uniform_480_50_2_1.dat`

## Instance Format

Each `.dat` file is a plain-text, space-separated file:

```
n m k
s_1[1] s_1[2] ... s_1[m]
s_2[1] s_2[2] ... s_2[m]
...
s_n[1] s_n[2] ... s_n[m]
```

- First line: `n` (number of sequences), `m` (sequence length), `k` (alphabet size)
- Next `n` lines: integer sequences with values in `{0, 1, ..., k-1}`

File naming convention: `{type}_{n}_{m}_{k}_{id}.dat` where type is `balanced` or `uniform`.

## Reproducibility

Both SA and HG2 accept a `--seed` flag for deterministic random number generation.
The SA cooling schedule is time-dependent (temperature updates based on wall-clock elapsed time), so results are deterministic on the **same machine, same Python version, and same system load**, but may differ across hardware due to varying iteration counts within the time budget. This is standard for time-limited metaheuristics.

## Usage
```bash
# Run Simulated Annealing on an instance
python main.py -a sa -i instances/uniform_480_50_2_2.dat --timeout 20

# Run HG2 (Hybrid Genetic Algorithm) on an instance
python main.py -a hg2 -i instances/uniform_480_50_2_2.dat --timeout 20

# Run exact solver (requires Gurobi) on an instance
python main.py -a gurobi -i instances/uniform_480_50_2_2.dat --timeout 20
```

## License
MIT License - see LICENSE file
