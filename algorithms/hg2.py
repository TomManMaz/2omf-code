"""
Hybrid Genetic Algorithm (HG2) - Hyper-Optimized Implementation

All operations use raw NumPy arrays. No Solution objects in hot paths.
Precomputes everything possible. Minimizes allocations in loops.

References:
    Dang, D.T., Nguyen, N.T. & Hwang, D. (2023)
    "Hybrid genetic algorithms for the determination of DNA motifs
    to satisfy postulate 2-Optimality"
    Applied Intelligence 53, 8644-8653
"""

import time
import numpy as np
from numpy.random import Generator, PCG64

from data.instance import Instance
from data.result import Result
from config.algorithm_params import HG2Config
import logging                                                                                                                                                                              
logger = logging.getLogger(__name__)   


class HG2:
    """
    Hyper-optimized HG2 using pure NumPy arrays.

    Population stored as 2D int32 array (population_size x seq_len).
    All genetic operators work directly on arrays.
    """

    __slots__ = ('population_size', 'num_offspring', 'num_paired_parents', 'mut_rate', 'num_elites', 'name')

    def __init__(self,
                 population_size: int = None,
                 num_offspring: int = None,
                 num_paired_parents: int = None,
                 mutation_rate: float = None,
                 num_elites: int = None) -> None:
        config = HG2Config
        self.population_size = population_size if population_size is not None else config.population_size
        self.num_offspring = num_offspring if num_offspring is not None else config.num_offspring
        self.num_paired_parents = num_paired_parents if num_paired_parents is not None else config.num_paired_parents
        self.mut_rate = mutation_rate if mutation_rate is not None else config.mutation_rate
        self.num_elites = num_elites if num_elites is not None else config.num_elites
        self.name = 'hg2_fast'

    def minimize_hg2(self, instance: Instance, timeout: float = 60.0,
                 disp: int = 0, seed: int = 42) -> Result:
        """Run HG2 optimization."""

        rng = Generator(PCG64(seed))
        t_start = time.perf_counter()

        # Instance data (read-only)
        S = instance.sequences                    # (n, m) int array
        n, m, k = instance.num_sequences, instance.sequence_length, instance.alphabet_size

        logger.info(f'Starting HG2Fast on instance {instance.name} with n={n}, m={m}, k={k}, timeout={timeout}s')
        logger.info(f'  population_size={self.population_size}, n_offspring={self.num_offspring}, num_paired_parents={self.num_paired_parents}, mut_rate={self.mut_rate:.4f}, self.num_elites={self.num_elites}')
        # Precompute column data for fast delta calculations
        # col_masks[j, a] = boolean mask where S[:, j] == a
        col_masks = np.zeros((m, k, n), dtype=np.bool_)
        for j in range(m):
            for a in range(k):
                col_masks[j, a] = (S[:, j] == a)

        # --- Helper functions (closures for speed) ---

        def eval_one(seq):
            """Evaluate single sequence -> objective value."""
            d = np.sum(S != seq, axis=1)  # hamming distances
            return np.sum(d * d)

        def eval_batch(seqs):
            """Evaluate multiple sequences -> array of objectives."""
            n_seq = seqs.shape[0]
            objs = np.empty(n_seq, dtype=np.int64)
            for i in range(n_seq):
                d = np.sum(S != seqs[i], axis=1)
                objs[i] = np.sum(d * d)
            return objs

        def get_distances(seq):
            """Get hamming distances from seq to all strings in S."""
            return np.sum(S != seq, axis=1)

        def calc_delta(seq, dists, pos, old_c, new_c):
            """Calculate objective change for single-position mutation."""
            m1 = col_masks[pos, old_c]  # strings matching old char
            m2 = col_masks[pos, new_c]  # strings matching new char
            # Strings gaining match: delta -= 2*(d-1)+1
            # Strings losing match: delta += 2*d+1
            delta = -np.sum(2 * (dists[m2] - 1) + 1) + np.sum(2 * dists[m1] + 1)
            return delta

        def bldc(seq):
            """Best Local Descent - single best improvement step."""
            seq = seq.copy()
            dists = get_distances(seq)
            best_delta, best_pos, best_char = 0, -1, -1

            for pos in range(m):
                old_c = seq[pos]
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    delta = calc_delta(seq, dists, pos, old_c, new_c)
                    if delta < best_delta:
                        best_delta, best_pos, best_char = delta, pos, new_c

            if best_pos >= 0:
                old_c = seq[best_pos]
                seq[best_pos] = best_char
                # Update distances
                dists[col_masks[best_pos, old_c]] += 1
                dists[col_masks[best_pos, best_char]] -= 1

            return seq, int(np.sum(dists * dists))

        def tournament(objs, size=3):
            """Tournament selection, returns winner index."""
            idx = rng.choice(len(objs), size=size, replace=False)
            return idx[np.argmin(objs[idx])]

        # --- Initialize population ---
        pop = rng.integers(0, k, size=(self.population_size, m), dtype=np.int32)
        objs = eval_batch(pop)

        best_seq = pop[np.argmin(objs)].copy()
        best_obj = int(np.min(objs))
        n_improv = 0
        gen = 0

        if disp:
            logger.info(f'HG2Fast: n={n}, m={m}, k={k}, timeout={timeout}s')
            logger.info(f'  pop={self.population_size}, offspring={self.num_offspring}, mut={self.mut_rate:.4f}')
            logger.info(f'  Initial best: {best_obj}')

        # --- Main loop ---
        while time.perf_counter() - t_start < timeout:
            gen += 1

            # 1) Select elites
            elite_idx = np.argpartition(objs, self.num_elites)[:self.num_elites]
            elites = pop[elite_idx].copy()

            # 2) IEBL: improve elites via local search
            improved = np.empty_like(elites)
            improved_obj = np.full(self.num_elites, np.iinfo(np.int64).max, dtype=np.int64)
            for i in range(self.num_elites):
                if time.perf_counter() - t_start >= timeout:
                    break
                improved[i], improved_obj[i] = bldc(elites[i])

            # Track best
            idx_best = np.argmin(improved_obj)
            if improved_obj[idx_best] < best_obj:
                best_obj = int(improved_obj[idx_best])
                best_seq = improved[idx_best].copy()
                n_improv += 1
                elapsed = time.perf_counter() - t_start
                logger.info(f'  Gen {gen}: NEW BEST = {best_obj}, time={elapsed:.2f}s')

            if time.perf_counter() - t_start >= timeout:
                break

            # 3) KLD: find most distant from improved elites
            non_elite_mask = np.ones(self.population_size, dtype=np.bool_)
            non_elite_mask[elite_idx] = False
            candidates = pop[non_elite_mask]

            distant = np.empty_like(improved)
            distant_obj = np.empty(self.num_elites, dtype=np.int64)
            if len(candidates) > 0:
                for i in range(self.num_elites):
                    dists_to_elite = np.sum(candidates != improved[i], axis=1)
                    far_idx = np.argmax(dists_to_elite)
                    distant[i] = candidates[far_idx]
                distant_obj = eval_batch(distant)
            else:
                distant = improved.copy()
                distant_obj = improved_obj.copy()

            # 4) Selection (tournament)
            parent_idx = np.empty((self.num_paired_parents, 2), dtype=np.int32)
            for i in range(self.num_paired_parents):
                parent_idx[i, 0] = tournament(objs)
                parent_idx[i, 1] = tournament(objs)
                while parent_idx[i, 1] == parent_idx[i, 0]:
                    parent_idx[i, 1] = tournament(objs)

            # 5) Crossover (two-point)
            offspring = np.empty((self.num_offspring, m), dtype=np.int32)
            for i in range(0, self.num_offspring, 2):
                pi = rng.integers(0, self.num_paired_parents)
                p1, p2 = parent_idx[pi]
                pts = np.sort(rng.choice(m, size=2, replace=False))

                c1 = pop[p1].copy()
                c2 = pop[p2].copy()
                c1[pts[0]:pts[1]] = pop[p2, pts[0]:pts[1]]
                c2[pts[0]:pts[1]] = pop[p1, pts[0]:pts[1]]

                offspring[i] = c1
                if i + 1 < self.num_offspring:
                    offspring[i + 1] = c2

            # 6) Mutation (triple-value operator from paper)
            # For each selected individual, create k-1 mutants (one for each alternative value)
            mut_mask = rng.random(self.num_offspring) < self.mut_rate
            mut_idx = np.where(mut_mask)[0]
            if len(mut_idx) > 0:
                mutants_list = []
                for i in mut_idx:
                    mut_pos = rng.integers(0, m)
                    old_val = offspring[i, mut_pos]
                    # Create one mutant for each alternative value at this position
                    for new_val in range(k):
                        if new_val != old_val:
                            mutant = offspring[i].copy()
                            mutant[mut_pos] = new_val
                            mutants_list.append(mutant)
                # Add all mutants to offspring pool (survival selection will handle size)
                if mutants_list:
                    mutants = np.array(mutants_list, dtype=np.int32)
                    offspring = np.vstack([offspring, mutants])

            # 7) Evaluate offspring
            off_objs = eval_batch(offspring)

            # 8) Survival selection
            # Combine non-elites + offspring, select best
            non_elite_pop = pop[non_elite_mask]
            non_elite_obj = objs[non_elite_mask]

            combined = np.vstack([non_elite_pop, offspring])
            combined_obj = np.concatenate([non_elite_obj, off_objs])

            n_survive = self.population_size - self.num_elites - self.num_elites
            surv_idx = np.argpartition(combined_obj, n_survive)[:n_survive]

            pop = np.vstack([combined[surv_idx], improved, distant])
            objs = np.concatenate([combined_obj[surv_idx], improved_obj, distant_obj])

            if disp and gen % 10 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(f'  Gen {gen}: best={np.min(objs)}, mean={np.mean(objs):.1f}, time={elapsed:.2f}s')

        # Final local search
        final_seq, final_obj = bldc(best_seq)
        if final_obj < best_obj:
            best_obj = final_obj
            best_seq = final_seq
            n_improv += 1

        duration = time.perf_counter() - t_start

        if disp:
            logger.info(f'HG2Fast done: obj={best_obj}, gen={gen}, improv={n_improv}, time={duration:.2f}s')

        return Result(
            x=best_seq.tolist(),
            objective_value=best_obj,
            duration=duration,
            n_iterations=gen,
            n_improvements=n_improv,
            message="OK",
            status=0
        )
