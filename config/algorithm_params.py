from dataclasses import dataclass

@dataclass(frozen=True)
class HG2Config:
    """
    HG2 parameters

    These default parameters were tuned by IRACE (4.3.c5b213a) on benchmark instances:
    - Number of strings = [480, 1020, 2040]
    - String length = [50, 100, 200, 300]
    - Alphabet size = [2, 4, 10]
    - Tuning budget: 180 evaluations
    - Objective: minimum objective value
    - Date 2026-01-23

    See: docs/tuning/hg2_irace_results.md for full methodology
    """
    population_size: int = 81
    num_offspring: int = 160
    num_paired_parents: int = 55
    mutation_rate: float = 0.3043
    num_elites: int = 4


@dataclass(frozen=True)
class SimulatedAnnealingConfig:
    """
    Simulated Annealing parameters

    These default parameters were tuned by IRACE (4.3.c5b213a) on benchmark instances:
    - Number of strings = [480, 1020, 2040]
    - String length = [50, 100, 200, 300]
    - Alphabet size = [2, 4, 10]
    - Tuning budget: 180 evaluations
    - Objective: minimum objective value
    - Date 2026-01-23

    See: docs/tuning/sa_irace_results.md for full methodology
    """

    initial_temperature: float = 970.08
    final_temperature: float = 0.8412