from __future__ import annotations
from typing import List

import numpy as np

from data.instance import Instance



class Solution():
    def __init__(self, instance: Instance, sequence: List[int] = None):
        self.instance: Instance = instance
        self.sequence: np.ndarray = sequence
        self.objective_value: int = None
        self._distances = None



    def evaluate(self) -> int:
        """
        Evaluates the objective value for the current solution by computing the sum of squared Hamming distances
        between the solution's sequence and each sequence in the instance.

        Returns:
            int: The computed objective value, which is the sum of squared Hamming distances.
        """

        # Compyute ALL Hamming distances in one vectorized operations
        self._distances = np.sum(self.instance.sequences != self.sequence, axis=1)
        self.objective_value = int(np.sum(self._distances ** 2))
        return self.objective_value

    def calculate_norm_1(self) -> int:
        """
        Computes the L1 norm of the solution's sequence, which is the sum of Hamming distances
        between the solution and each sequence in the instance.

        Returns:
            int: The L1 norm of the solution's sequence.
        """
        return int(self._distances.sum())

    def calculate_objective_delta(self, index: int, new_character: int) -> int:
        """
        Compute the change (delta) in the objective value when the character at a
        given column index of the current solution sequence is replaced with a new
        character.

        Args:
            index (int): Column index in the sequence to change.
            new_character (int): The new character value to place at the given index.

        Returns:
            int: The objective difference (new_objective - current_objective) resulting
            from performing the single-character replacement.

        Behavior:
            - Uses self.instance.sequences[:, index] to obtain the column of characters
              across all sequences.
            - Compares that column to the current character (self.sequence[index]) and
              to the proposed new_character to build two boolean masks.
            - Sums precomputed pairwise distance contributions in self._distances for
              rows matching each mask.
            - Returns the value computed as:
                2 * (-sum_dist_s2 + sum_dist_s1) + count(mask_s2) + count(mask_s1)
              where sum_dist_s1/s2 are the summed distances for matches to the current
              and new characters respectively, and count(...) is the number of matches.
              (The factor 2 accounts for symmetric pair contributions in the objective.)

        Notes:
            - Assumes self.instance.sequences is a 2D array-like (num_sequences x num_columns),
              self.sequence is the current solution row, and self._distances is an array
              of precomputed distance contributions aligned with the sequence rows.
            - Complexity is O(num_sequences) due to column comparisons and masked sums.
        """
        column = self.instance.sequences[:, index]
        current_character = self.sequence[index]

        mask_s1 = (column == current_character)
        mask_s2 = (column == new_character)

        sum_dist_s1 = int(self._distances[mask_s1].sum())
        sum_dist_s2 = int(self._distances[mask_s2].sum())

        return 2 * (-sum_dist_s2 + sum_dist_s1) + int(mask_s2.sum()) + int(mask_s1.sum())

    def copy(self) -> Solution:
        new_sol = Solution(self.instance)
        new_sol.sequence = self.sequence.copy()
        new_sol.objective_value = self.objective_value
        new_sol._distances = self._distances.copy() if self._distances is not None else None
        return new_sol

    def to_file(self, file_name: str) -> None:
        """
        Write the solution to a file.
        
        Parameters
        ----------
        file_name : str
            The name of the file where the solution will be written."""
        if self.sequence is None:
            raise ValueError("Solution sequence is not set. Please set the sequence before writing to a file.")
        with open(file_name, 'w') as f:
            f.write(' '.join(map(str, self.sequence)))


    def get_mode_lb(self) -> float:
        """
        Calculates a lower bound for the mode of the solution's sequence.
        Returns:
            float: The computed lower bound for the mode, calculated as the square of the L1 norm of the sequence
                   divided by the number of sequences in the instance.
        Raises:
            ValueError: If the solution has not been initialized with a sequence (i.e., if `self.sequence` is None).
        """
        if self.sequence is None:
            raise ValueError("Solution has not been initialized with a sequence. Call evaluate() first.")
            
        norm_1 = self.calculate_norm_1() 
        return norm_1**2/self.instance.num_sequences
        


        
        
        