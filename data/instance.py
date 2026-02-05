from __future__ import annotations
import random 
from typing import List
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform



class Instance:
    # an instance of the 2omf problem is a set of n sequences of length m over an alphabet of size k

    def __init__(self, 
                 sequence_length: int, 
                 num_sequences: int, 
                 alphabet_size: int, 
                 sequences: List[List[int]]| np.ndarray):
        self.sequence_length: int = sequence_length
        self.num_sequences: int = num_sequences
        self.alphabet_size: int = alphabet_size
        self.sequences: np.ndarray = np.array(sequences, dtype=int)
        self.name = None

    def copy(self) -> Instance:
        new_strings = self.sequences.copy()
        return Instance(self.sequence_length, self.num_sequences, self.alphabet_size, new_strings)

    @staticmethod
    def from_list(strings: List[List[int]]) -> Instance:
        """
        Creates an Instance object from a list of integer sequences.

        Args:
            strings (List[List[int]]): A list of sequences, where each sequence is a list of integers. All sequences must have the same length.

        Returns:
            Instance: An Instance object initialized with the parameters derived from strings.

        Raises:
            ValueError: If the sequences in strings do not all have the same length.
        """
        letters = set()
        for string in strings:
            for letter in string:
                letters.add(letter)
        k = len(letters)
        m = len(strings[0])
        n = len(strings)

           # Validate that all strings have the same length
        if not all(len(s) == m for s in strings):
            raise ValueError("All sequences must have the same length.")

        strings_np = np.array(strings, dtype=int)
        return Instance(m, n, k, strings_np)

    @staticmethod
    def from_file(filename: str) -> Instance:
        path = Path(filename)
        with open(filename, 'r') as f:
            n, m, k = [int(i) for i in f.readline().split()]
            sequences = []
            for _ in range(n):
                sequences.append([int(i) for i in f.readline().split()])
        instance = Instance(m, n, k, sequences)
        instance.name = path.stem
        return instance
    
    def print_to_file(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(f'{self.num_sequences} {self.sequence_length} {self.alphabet_size}\n')
            for string in self.sequences:
                f.write(' '.join([str(i) for i in string]) + '\n')
    
    @staticmethod
    def from_probability(n_of_sequence: int, m:int, p: float) -> Instance:
        """
        Generate a binary instance with probability of having 0 in any position equalts to p 

        Parameters
        ----------
        n_of_sequence : int
            Number of sequences to generate.
        m : int
            Length of the sequences.
        p : float
            Probability of having 0 in any position.

        Returns
        -------
        Instance: Instance
            An instance of the 2OMF problem.
        """
        profile = []
        for _ in range(n_of_sequence):
            profile.append([0 if random.random() < p else 1 for _ in range(m)])
        return Instance.from_list(profile)

    @staticmethod
    def from_distribution(n_of_sequence: int, distribution: List[List[float]]) -> Instance:
        """
        Generate an instance from a distribution.
        
        Parameters
        ----------
        n_of_sequence : int
            Number of sequences to generate.
        distribution : List[List[float]]
            A matrix of size m x k, where m is the length of the sequences and k is the size of the alphabet.
            Each row of the matrix is a probability distribution over the alphabet.

        Returns
        -------
        Instance: Instance
            An instance of the 2OMF problem.
        """
        profile = []
        m = len(distribution)
        alphabet = [i for i in range(len(distribution[0]))]
        profile = [[random.choices(alphabet, weights=distribution[j])[0] for j in range(m)]for _ in range(n_of_sequence)]
        return Instance.from_list(profile)

    def create_sub_instance(self, indices: List[int]) -> Instance:
        """
        Create a sub-instance from the original instance by selecting the components of the sequences with the given indices.
        If the original instance has n sequences of length m , the sub-instance will have n sequences with length len(indices). 

        Parameters
        ----------
        indices : List[int]
            List of indices to select.

        Returns
        -------
        Instance: Instance
            An instance of the 2OMF problem.
        """
    

        sub_instance = self.sequences[:, indices]
        return Instance(len(indices), self.num_sequences, self.alphabet_size, sub_instance) 
        
    @staticmethod
    def generate_uniform(n: int, m: int, k: int, p: float = 0.5) -> Instance:
        """
        Generate an instance from a uniform distribution.
        For binary alphabets (k==2) this uses a binomial draw:
            np.random.binomial(1, 1-p, size=(n, m))
        where `p` is the probability of a 0 (so 1-p is the probability of a 1).

        Parameters
        ----------
        n : int
            Number of sequences to generate.
        m : int
            Length of the sequences.
        k : int
            Size of the alphabet.
        p : float, optional
            For k==2, probability of 0 in each position (default 0.5).
        """

        if k == 2:
            arr = np.random.binomial(1, 1 - p, size=(n, m)).astype(int)
        else:
            rng = np.random.default_rng()
            arr = rng.integers(0, k, size=(n, m), dtype=int)

        return Instance(m, n, k, arr)

  
    @staticmethod
    def generate_balanced(n: int, m: int, k: int) -> Instance:
        """
        Generates an instance of the 2OMF(n, m, k) in this way:
        - First of all, check if k divides n. If not, raise an error.
        - Creates a list of columns, each containing with a balanced number of letters: n/k 0's, 
        - Shuffle the columns

        Parameters
        ----------

        n : int
            Number of strings in the instance

        m : int
            Length of the strings

        k : int
            Size of the alphabet

        Returns
        -------

        Instance: Instance
            The resulting instance of the 2OMF problem.

        Raises
        ------
        TypeError
            If n, m, or k are not integers.
        ValueError
            If n, m, or k are not positive.
            If n is not divisible by k.
        """
        # 2. Divisibility Check 
        if n % k != 0:
            raise ValueError(f'Error: n={n} is not divisible by k={k}')
        
        # 3. Calculate block_length 
        block_length = n // k

        # 4. Create the elementary column
        elementary_column = []
        for block_val in range(k): 
            constant_list = [block_val for _ in range(block_length)]
            elementary_column.extend(constant_list)
        
        # 5. Generate and shuffle columns
        columns = []
        for _ in range(m): # Use _ for loop variable if not explicitly used
            col = elementary_column.copy()
            random.shuffle(col)
            columns.append(col)


        # 6. Transpose from m x n to n x m and create Instance
        strings_list = [list(row) for row in zip(*columns)]
        return Instance.from_list(strings_list)

    def get_random_string(self) -> List[int]:
        return random.choice(self.sequences)
    
    def get_two_random_strings(self) -> List[List[int]]:
        return random.sample(self.sequences, 2)

    def get_random_sample(self, sample_size: int) -> List[List[int]]:
        return random.sample(self.sequences, sample_size)
    
    def get_pairwise_distances(self):
        """
        Calculate pairwise distances between points in the strings array using the Hamming distance metric.
               
        Returns
        -------

            numpy.ndarray: An array of pairwise distances.
        """
        
        x = np.array(self.sequences)
        return pdist(X=x, metric='hamming')
    
    def get_distance_matrix(self):
        """
        Calculate the distance matrix between points in the strings array using the Hamming distance metric.

        Returns
        -------

            numpy.ndarray: A distance matrix.
        """
        x = self.get_pairwise_distances()
        return squareform(x)
    
    def get_mode_bound(self) -> int:
        """
        Vectorized version: for each column j compute u_j = n - sum(column_j) and return sum(u_j) // m
        """
        arr = np.asarray(self.sequences, dtype=int)  # shape (n, m)
        col_sums = arr.sum(axis=0)                   # shape (m,)
        u_total = int((self.num_sequences - col_sums).sum())
        return u_total // self.sequence_length
    
    @property
    def mode_bound(self) -> int:
        """
        Calculate the mode bound of the instance.
        
        Returns
        -------
        int: The mode bound of the instance.
        """
        return self.get_mode_bound()


    def get_static_bound(self) -> int:
        """
        Computes a static lower bound (lb) based on the mismatch distribution of strings over positions and alphabet letters.

        The method analyzes, for each position and each letter in the alphabet, how many strings do not match that letter at that position.
        For each position, it finds the minimum number of mismatches across all letters (the "mode" per position).
        It then sums these minimum mismatches across all positions to get an initial lower bound of unmatched positions (lb_unmatches).
        The method iteratively distributes these unmatched positions across fill-up levels, incrementally increasing the lower bound (lb)
        according to a specific weighting formula until all unmatched positions are accounted for.

        Returns:
            int: The computed static lower bound.
        """
        n = self.num_sequences                                                                                                                                                                  
        m = self.sequence_length                                                                                                                                                                
        k = self.alphabet_size                                                                                                                                                                  
                                                                                                                                                                                                
        # Use bincount for each column - faster than broadcasting for large k                                                                                                                   
        counts = np.array([np.bincount(self.sequences[:, j], minlength=k)                                                                                                                       
                            for j in range(m)])  # shape (m, k)                                                                                                                                  
                                                                                                                                                                                                
        mode_per_position = n - counts.max(axis=1)                                                                                                                                              
        lb_unmatches = int(mode_per_position.sum())      

        
        fill_up_level = 0
        lb = 0

        while lb_unmatches > 0:
            level_unmatches = min(self.num_sequences, lb_unmatches)
            lb_unmatches -= level_unmatches
            if level_unmatches > 0:
                lb += ((2 * fill_up_level) + 1) * level_unmatches

            fill_up_level += 1

    

        return lb

    @property
    def static_bound(self) -> int:
        """
        Calculate the static bound of the instance.
        
        Returns
        -------
        int: The static bound of the instance.
        """
        return self.get_static_bound()

    @staticmethod
    def generate_balanced_with_patterns(n:int, m:int, k:int, n_patterns:int) -> Instance:
        """
        Generates an instance of the 2OMF(n, m, k) in this way:
        - First of all, check if k divides n. If not, raise an error.
        - Creates a list of columns, each containing with a balanced number of letters: n/k 0's,
        - Shuffle the columns
        Parameters
        ----------
        n : int
            Number of strings in the instance. Must be divisible by k.
        m : int
            Length of the strings
        k : int
            Size of the alphabet
        n_patterns : int    
            Number of patterns to generate. 
        Returns
        -------
        Instance: Instance
            The resulting instance of the 2OMF problem.
        """
        
        if n % k != 0:
            raise ValueError(f'Error: n={n} is not divisible by k={k}')
   
        block_length = n // k
        elementary_column = []
        for block in range(k):
            constant_list = [block for _ in range(block_length)]
            elementary_column.extend(constant_list)

        # consider n_patterns shuffled columns
        patterns = []
        for i in range(n_patterns):
            col = elementary_column.copy()
            random.shuffle(col)
            patterns.append(col)


        # generate m random columns sampling from the patterns
        columns = []
        for i in range(m):
            col = random.choice(patterns)
            columns.append(col)


        lista = list(zip(*columns))
        return Instance.from_list([list(row) for row in lista])
    


    def get_associated_matrix(self) -> np.ndarray:
        """
        Create an associated matrix for the instance.
        
        The associated matrix is a binary matrix where each row corresponds to a sequence in the instance,
        and each column corresponds to a position in the sequence. The value is 1 if the letter at that position
        matches the letter in the alphabet, and 0 otherwise.

        Returns
        -------
        np.ndarray: The associated matrix.
        """
        n = self.num_sequences
        m = self.sequence_length
        k = self.alphabet_size
        matrix = np.zeros((n, m * k), dtype=int)
        
        col_offsets = np.arange(m) * k
        col_indices = self.sequences + col_offsets  # Broadcasting: (n, m) + (m,)

        matrix = np.zeros((n, m * k), dtype=int)
        row_indices = np.arange(n)[:, None]  # Shape (n, 1) for broadcasting
        matrix[row_indices, col_indices] = 1
        return matrix


    def get_smallest_nonzero_singularvalue(self) -> float:
        """
        Calculate the smallest non-zero singular value of the associated matrix.
        
        Returns
        -------
        float: The smallest non-zero singular value of the associated matrix.
        """
        associated_matrix = self.get_associated_matrix()
        # _, s, _ = np.linalg.svd(associated_matrix, full_matrices=False)
        eigenvalues, eigenvectors = np.linalg.eigh(associated_matrix.T @ associated_matrix)
        # s = np.sort(s)
        smallest_nonzero_singular_value = eigenvalues[self.sequence_length -1]
        # print 1 eigenvector [ the one corresponding to the smallest non-zero singular value]

        print("Eigenvector", eigenvectors[:, self.sequence_length - 1])
        print("Eigenvalue", smallest_nonzero_singular_value)
        return smallest_nonzero_singular_value
    

    def get_eigenvalue_bound(self) -> float:
        """
        Calculate the eigenvalue bound of the instance.
        
        The eigenvalue bound is calculated as the smallest non-zero singular value of the associated matrix.
        
        Returns
        -------
        float: The eigenvalue bound of the instance.
        """
        nonzero_singular_value =  self.get_smallest_nonzero_singularvalue()
        eigenvalue_bound = nonzero_singular_value *  (self.sequence_length * self.alphabet_size - self.sequence_length)
        return eigenvalue_bound

    @staticmethod
    def generate_bad_lower_bound(n: int, m: int, k: int, cluster_ratio: float = 0.5) -> Instance:
        """
        Generate an instance where the 1-OMF solution is significantly worse than the 2-OMF solution,
        creating a large gap between lower bounds.

        The strategy is to create two distinct clusters of sequences where:
        - Each cluster has sequences that are similar to each other
        - The two clusters are maximally different from each other
        - A single median string can only serve one cluster well
        - Two median strings (one per cluster) provide a much better solution

        Parameters
        ----------
        n : int
            Number of sequences to generate. Should be even for best results.
        m : int
            Length of the sequences.
        k : int
            Size of the alphabet. Should be at least 2.
        cluster_ratio : float, optional
            Ratio of positions that define cluster separation (default 0.5).
            Higher values create more distinct clusters.

        Returns
        -------
        Instance
            An instance with a bad lower bound (large gap between 1-OMF and 2-OMF).

        Raises
        ------
        ValueError
            If k < 2 or cluster_ratio not in (0, 1).
        """
        if k < 2:
            raise ValueError("Alphabet size k must be at least 2")
        if not 0 < cluster_ratio < 1:
            raise ValueError("cluster_ratio must be between 0 and 1")

        # Split sequences into two equal clusters
        n_cluster = n // 2

        # Determine how many positions separate the clusters
        m_separator = int(m * cluster_ratio)
        m_common = m - m_separator

        sequences = []

        # Cluster 1: Use letter 0 for separator positions, random for common positions
        for _ in range(n_cluster):
            seq = []
            # Separator positions - mostly letter 0 with small noise
            for _ in range(m_separator):
                if random.random() < 0.9:  # 90% consistency within cluster
                    seq.append(0)
                else:
                    seq.append(random.randint(0, k - 1))
            # Common positions - random
            for _ in range(m_common):
                seq.append(random.randint(0, k - 1))
            sequences.append(seq)

        # Cluster 2: Use letter 1 for separator positions, random for common positions
        for _ in range(n - n_cluster):  # Handle odd n
            seq = []
            # Separator positions - mostly letter 1 with small noise
            for _ in range(m_separator):
                if random.random() < 0.9:  # 90% consistency within cluster
                    seq.append(min(1, k - 1))
                else:
                    seq.append(random.randint(0, k - 1))
            # Common positions - random
            for _ in range(m_common):
                seq.append(random.randint(0, k - 1))
            sequences.append(seq)

        # Shuffle sequences to mix clusters
        random.shuffle(sequences)

        return Instance.from_list(sequences)



    @staticmethod
    def generate_clustered(n: int, m: int, k: int, cluster_ratio: float = 0.05) -> Instance:
        if k!= 2:
            raise ValueError("Currently only binary alphabet (k=2) is supported for clustered instance generation.")
        profile_0 = [ [0 if random.random() > cluster_ratio else 1 for _ in range(m)] for _ in range(n//2)]

        profile_1 = []
        profile_0_np = np.array(profile_0)
        for index in range(m): # Use _ for loop variable if not explicitly used

            col = 1-profile_0_np[:, index]
            random.shuffle(col)
            col = col.tolist()
            profile_1.append(col)


        profile_1_transposed = [list(row) for row in zip(*profile_1)]
        profile = profile_0 + profile_1_transposed

        return Instance.from_list(profile)

    @staticmethod
    def generate_bad_static_bound(n: int, m: int, k: int = 2, noise_rate: float = 0.05) -> Instance:
        """
        Generate instances where static_bound is far from the optimal objective value.

        Creates two tight clusters that are approximate complements of each other:
        - Cluster 1: mostly 0s with small noise
        - Cluster 2: mostly 1s with small noise

        This results in:
        - Columns that are approximately balanced (n/2 zeros, n/2 ones)
        - Static bound sees high minimum mismatches per position
        - But optimal solution uses one median per cluster with very low total distance

        Parameters
        ----------
        n : int
            Number of sequences to generate. Should be even for best results.
        m : int
            Length of the sequences.
        k : int, optional
            Size of the alphabet. Currently only k=2 is supported (default 2).
        noise_rate : float, optional
            Probability of flipping a bit from the cluster center (default 0.05).
            Lower noise_rate creates tighter clusters and larger gap.

        Returns
        -------
        Instance
            An instance where static_bound >> optimal objective value.
            Expected gap ratio: approximately m / (4 * noise_rate)

        Raises
        ------
        ValueError
            If k != 2 or noise_rate not in (0, 1).
        """
        if k != 2:
            raise ValueError("Currently only binary alphabet (k=2) is supported for bad_static_bound generation.")
        if not 0 < noise_rate < 1:
            raise ValueError("noise_rate must be between 0 and 1")

        # Generate cluster 1: mostly 0s with small noise
        cluster1 = []
        for _ in range(n // 2):
            seq = [0 if random.random() > noise_rate else 1 for _ in range(m)]
            cluster1.append(seq)

        # Generate cluster 2: mostly 1s with small noise
        cluster2 = []
        for _ in range(n - n // 2):
            seq = [1 if random.random() > noise_rate else 0 for _ in range(m)]
            cluster2.append(seq)

        # Combine and shuffle to mix clusters
        sequences = cluster1 + cluster2
        random.shuffle(sequences)

        return Instance.from_list(sequences)
