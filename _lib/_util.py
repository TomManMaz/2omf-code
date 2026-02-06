from typing import Optional, List
from datetime import date
from pathlib import Path
import csv
import numpy as np

from utils.logging import get_logger
from classes.instance import Instance

logger = get_logger(__name__)

NA_STRING = "N/A"

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}

def _dict_formatter(d, n=0, mplus=1, sorter=None):
    """
    Pretty printer for dictionaries

    `n` keeps track of the starting indentation;
    lines are indented by this much after a line break.
    `mplus` is additional left padding applied to keys
    """
    if isinstance(d, dict):
        m = max(map(len, list(d.keys()))) + mplus  # width to print keys
        s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                       _indenter(_dict_formatter(v, m+n+2, 0, sorter), m+2)
                       for k, v in sorter(d)])  # +2 for ': '
    else:
        # By default, NumPy arrays print with linewidth=76. `n` is
        # the indent at which a line begins printing, so it is subtracted
        # from the default to avoid exceeding 76 characters total.
        # `edgeitems` is the number of elements to include before and after
        # ellipses when arrays are not shown in full.
        # `threshold` is the maximum number of elements for which an
        # array is shown in full.
        # These values tend to work well for use with OptimizeResult.
        with np.printoptions(linewidth=76-n, edgeitems=2, threshold=12,
                             formatter={'float_kind': _float_formatter_10}):
            s = str(d)
    return s



class _NiceResult:
    def __init__(self, 
        duration: float,
        x: List[int],
        objective_value: float,
        n_iterations: int,
        status: int = 0,
        message: Optional[str] = None,
        n_improvements: Optional[int] = None,
        n_nodes: Optional[int] = None,
        gap: Optional[float] = None,
        lower_bound: Optional[float]=None) -> None:

        self.duration = duration
        self.x = x
        self.objective_value = objective_value
        self.n_iterations = n_iterations
        self.message = message if message is not None else _status_message['success']
        self.n_improvements = n_improvements    
        self.n_nodes = n_nodes
        self.gap = gap
        self.lower_bound = lower_bound

    



    
    
    def log_result(self,algorithm_name: str) -> None:
        """
        Logs the results of an algorithm execution, including objective function value, number of iterations,
        nodes explored (if available), number of improvements (if available), and duration.

        Args:
            algorithm_name (str): The name of the algorithm whose results are being logged.

        Returns:
            None
        """
        logger.info(f'===========================================')
        logger.info(f'Algorithm: {algorithm_name}')
        logger.info(f'-------------------------------------------')
        # logger.info(f"{self.message}")
        logger.info(f'\tObjective function value: {self.objective_value}')
        # logger.info(f"         Iterations: {k:d}")
        logger.info(f'\tIterations: {self.n_iterations}')
        if self.n_nodes is not None:
            logger.info(f'\tNodes explored: {self.n_nodes}')
        if self.n_improvements is not None:
            logger.info(f'\tImprovements: {self.n_improvements}')
        if self.gap is not None:
            logger.info(f'\tMIP optimality gap: {self.gap:.2%}')
        if self.lower_bound is not None:
            logger.info(f'\tLower bound: {self.lower_bound:.2f}')
        logger.info(f'\tDuration: {self.duration:.2f} seconds')
        logger.info(f'===========================================')


    def to_file(self, seed: int, instance: Instance, algorithm_name: str, result_file: Path) -> None:        
        """
        Appends the current result data to a CSV file. If the file does not exist, writes the header first.

        Parameters:
            seed (int): The random seed used for the experiment.
            instance_name (str): The name of the problem instance.
            algorithm_name (str): The name of the algorithm used.
            result_file (Path): The path to the CSV file where results will be stored.

        The following fields are written to the CSV:
            - date: The current date in YYYY-MM-DD format.
            - Instances: The name of the problem instance.
            - algorithm: The name of the algorithm.
            - num_sequences: The number of sequences in the instance.
            - sequence_length: The length of each sequence in the instance.
            - alphabet_size: The size of the alphabet used in the instance.
            - seed: The random seed.
            - duration: The duration of the run.
            - objective_value: The objective function value.
            - iterations: The number of iterations (or "N/A" if not available).
            - improvements: The number of improvements (or "N/A" if not available).
            - gap: The optimality gap (or "N/A" if not available).
            - lb: The lower bound (or "N/A" if not available).
            - nodes: The number of nodes explored (or "N/A" if not available).
            - solution: The solution vector, joined as a semicolon-separated string.
        """

        _CSV_HEADER = (
            "date", "Instances","num_sequences","sequence_length","alphabet_size","algorithm", "seed", "duration", "objective_value", 
            "iterations", "improvements", "gap", "lb","nodes", "solution"
        )
        


        file_exists = result_file.exists()
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists: 
                writer.writerow(_CSV_HEADER)
            output = [
                date.today().strftime("%Y-%m-%d"),
                instance.name,
                instance.num_sequences,
                instance.sequence_length,
                instance.alphabet_size,
                algorithm_name, 
                seed,        
                self.duration,
                self.objective_value,
                self.n_iterations if self.n_iterations is not None else NA_STRING,
                self.n_improvements if self.n_improvements is not None else NA_STRING,
                self.gap if self.gap is not None else NA_STRING,
                self.lower_bound if self.lower_bound is not None else NA_STRING,
                self.n_nodes if self.n_nodes is not None else NA_STRING,
                ";".join(map(str, self.x))
            ]
            writer.writerow(output)  # Write the output as a row in the CSV file