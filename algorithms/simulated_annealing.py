# Simulated Annealing Algorithm Implementation
# Author: Felix Winter, Tommaso Mannelli Mazzoli

"""
A Simulated Annealing algorithm implementation for the 2-optimality motiv finding problem.
"""

import time
import random
import math

import numpy as np

from data.solution import Solution
from data.instance import Instance
from data.result import Result
from config.algorithm_params import SimulatedAnnealingConfig

from utils.logging import get_logger
from _lib._util import _status_message


__all__ = ["SimulatedAnnealing"]

logger = get_logger(__name__)

class SimulatedAnnealing:
    
    def __init__(self, initial_temperature: float=None, final_temperature: float=None):
        self.initial_temperature = initial_temperature if initial_temperature is not None else SimulatedAnnealingConfig.initial_temperature
        self.final_temperature = final_temperature if final_temperature is not None else SimulatedAnnealingConfig.final_temperature
        
    def _initialize_solution(self, instance: Instance) -> Solution:
        """
        Initialize a random solution for the given instance.
        """
        sequence = [random.randint(0, instance.alphabet_size - 1) for _ in range(instance.sequence_length)]
        return Solution(instance, sequence)

    def _calculate_cost_delta(self,old_letter: int, 
                              new_letter: int, 
                              position_idx: int,
                              match_indices_by_pos_char: list, 
                              mismatch_counts_per_string: list) -> int:
        """
        Calculates the change in cost (delta) when replacing a letter at a specific position
        in a sequence, considering matched and unmatched string indices.
        Args:
            old_letter (int): The integer representation of the letter currently at the position.
            new_letter (int): The integer representation of the letter to be placed at the position.
            position_idx (int): The index of the position in the sequence being modified.
            match_indices_by_pos_char (list): A nested list where match_indices_by_pos_char[pos][char]
                contains indices of strings that match character 'char' at position 'pos'.
            mismatch_counts_per_string (list): A list where each element is the current mismatch count
                for the corresponding string.
        Returns:
            int: The net change in cost resulting from replacing old_letter with new_letter at position_idx.
        """        
        delta = 0
        # 1) subtract all costs that were unmatched and are now matched
        for string_idx in match_indices_by_pos_char[position_idx][new_letter]:
            delta -= 2*(mismatch_counts_per_string[string_idx]-1)+1
        # 2) add all costs that are now unmatched
        for string_idx in match_indices_by_pos_char[position_idx][old_letter]:
            delta += 2*mismatch_counts_per_string[string_idx]+1
        return delta
    
    def _update_temperature(self, current_temp: float, iteration_count: int, elapsed_time: float, timeout: float) -> float:
        """
        Updates the temperature for the simulated annealing process based on the current temperature,
        iteration count, elapsed time, and total timeout.

        This method estimates the number of remaining iterations using the elapsed time and total timeout,
        then computes a cooling rate to gradually decrease the temperature towards the final temperature.

        Args:
            current_temp (float): The current temperature of the system.
            iteration_count (int): The number of iterations completed so far.
            elapsed_time (float): The elapsed time since the start of the process (in seconds).
            timeout (float): The total allowed time for the process (in seconds).

        Returns:
            float: The updated temperature for the next iteration.
        """

        if elapsed_time == 0:
            return current_temp
        remaining_iterations = (iteration_count / elapsed_time) * (timeout - elapsed_time)
        if remaining_iterations <= 0:
            return self.final_temperature
        cooling_rate = (self.final_temperature / current_temp) ** (1 / remaining_iterations)
        return current_temp * cooling_rate

    def _minimize_sa(self, instance: Instance, timeout: float = 60, disp:int=0, seed:int=42) -> Result:
        """
        Applies the Simulated Annealing algorithm to find an optimized solution for the given instance.

        Args:
            instance (Instance): The problem instance containing the sequences, alphabet size, and other relevant data.
            timeout (float, optional): The maximum time allowed for the algorithm to run, in seconds. Defaults to 60.

        Returns:
            result (Result): The optimization result represented as a "Result" object. Important attributes are: ''x'' the solution sequence,
            ''fun'' the value of the objective function at that solution, and ''message'' which describes the cause of the termination.
            See `data/result.py` for more details.
        Algorithm Overview:
            - Initializes a random solution and caches for mismatch counts.
            - Iteratively perturbs the solution by changing a random position to a new letter.
            - Accepts or rejects changes based on the cost difference and current temperature.
            - Updates the best solution found if a better one is discovered.
            - Gradually cools down the temperature according to a cooling schedule.
            - Stops when the timeout is reached, returning the best solution found.

        Notes:
            - Uses precomputed match/mismatch indices for efficient cost updates.
            - Logs progress and improvements during the search.
        """

        random.seed(seed)
        logger.info(f'Starting Simulated Annealing on instance {instance.name} with timeout {timeout} seconds with seed {seed}.')
        logger.info(f'Initial Temperature: {self.initial_temperature}, Final Temperature: {self.final_temperature}')
        start_time = time.perf_counter()
        alphabet_size = instance.alphabet_size
        num_strings = instance.num_sequences
        string_length = instance.sequence_length
        strings = instance.sequences
        message = []
   

        positions = range(string_length)
        string_indices = range(num_strings)
        alphabet = range(alphabet_size)
        mismatch_indices_by_pos_char = [
            [[s_no for s_no in string_indices if strings[s_no][p] != a] for a in alphabet] for p in
            positions]
        match_indices_by_pos_char = [
            [[s_no for s_no in string_indices if strings[s_no][p] == a] for a in alphabet] for p in
            positions]


        # init temp
        temp = self.initial_temperature

        # init random solution
        current_solution = self._initialize_solution(instance)

        # init unmatched letters per string cache
        mismatch_counts_per_string = [0] * num_strings
        for position_idx in positions:
            letter_offset = current_solution.sequence[position_idx]
            for string_idx in mismatch_indices_by_pos_char[position_idx][letter_offset]:
                mismatch_counts_per_string[string_idx] += 1

        # init cost cache
        current_cost = sum(count**2 for count in mismatch_counts_per_string)

        best_sol = current_solution.copy()
        best_cost = current_cost    
        logger.info(f"Initial solution with cost: {best_cost}")
        iteration_count = 1
        n_improvements = 0


        while time.perf_counter() - start_time < timeout:
            # select random position and letter to change
            position_idx = random.randint(0, string_length-1)
            offset = random.randint(1, alphabet_size-1)

            # update solution
            old_letter = current_solution.sequence[position_idx]
            new_letter = (old_letter + offset) % alphabet_size
            current_solution.sequence[position_idx] = new_letter

            cost_delta = self._calculate_cost_delta(old_letter, new_letter, position_idx,
                                                    match_indices_by_pos_char, mismatch_counts_per_string)
            new_cost = current_cost +  cost_delta

            if cost_delta < 0:
                accept = True
            elif temp > 0:
                accept =  random.random() < math.exp(-cost_delta / temp)
            else:
                accept = False # if temperature is zero, only accept if cost_delta is negative
            if accept:
                current_cost = new_cost

                # 1) update unmatches that are now matched
                for s_no in match_indices_by_pos_char[position_idx][new_letter]:
                    mismatch_counts_per_string[s_no] -= 1
                # 2) update new unmatches
                for s_no in match_indices_by_pos_char[position_idx][old_letter]:
                    mismatch_counts_per_string[s_no] += 1
            else:
                current_solution.sequence[position_idx] = old_letter

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time < timeout and current_cost < best_cost:
                n_improvements += 1
                best_sol = current_solution.copy()
                best_cost = current_cost
                logger.info(f"ETA: {elapsed_time:.2f} s\t Temperature: {temp:.2f} \t Found new best solution with cost: {best_cost}")

            # update temperature
            temp = self._update_temperature(temp, iteration_count, elapsed_time, timeout)
            iteration_count += 1

        if time.perf_counter() - start_time >= timeout:
            warnflag = 2
            msg = _status_message['maxiter']
            logger.info(f'SA done: obj={best_cost}, iters={iteration_count:,}, improv={n_improvements}, time={time.perf_counter() - start_time:.2f}s')
            logger.info(f'Timeout reached after {time.perf_counter() - start_time :.2f} seconds')
        # Setting the Result values
        res = Result(
            x=best_sol.sequence,
            objective_value=best_cost,
            duration=time.perf_counter() - start_time,
            n_iterations=iteration_count,
            n_improvements=n_improvements,
            message=msg if 'msg' in locals() else "Optimization completed successfully.",
            status=warnflag if 'warnflag' in locals() else 0
        )
        return res


