# Based on Gurobi 13.0.0
# Documentation gurobi = https://docs.gurobi.com/projects/optimizer/en/current/reference/python.html
from __future__ import annotations 

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import gurobipy as gp
    from gurobipy import GRB
else:
    try:
        import gurobipy as gp
        from gurobipy import GRB
        GUROBI_AVAILABLE = True
    except ImportError:
        GUROBI_AVAILABLE = False
        gp = None
        GRB = None

from data.instance import Instance
from data.solution import Solution
from data.result import Result
from utils.logging import get_logger
logger = get_logger(__name__)   


class OptimizationError(Exception):
    """Raised when Gurobi optimization fails."""
    pass


class GurobiWrapper():

    def __init__(self,   n_of_threads: int=1, gap_tolerance: float=0.0,
                 presolve: int=-1, preqlinearize: int=-1) -> None:
    
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is not installed. Please install Gurobi to use this class.")
        
        self.verbosity_level =0
        self.seed = 42
        self.name = 'gurobi'
        self.threads = n_of_threads
        self.gap_tolerance = gap_tolerance
        self.presolve = presolve
        self.preqlinearize = preqlinearize
        self.time_limit = None  

        

    def _get_solution(self, instance: Instance, x_vars: dict) -> Solution:
        """
        Extracts the solution from a Gurobi model after optimization.

        Args:
            model (gp.Model): The Gurobi model containing the optimized variables.
            instance (Instance): The problem instance containing string length and alphabet size.
            x_vars (dict): Dictionary mapping (i, j) tuples to Gurobi variables, where i is the position in the string and j is the alphabet index.

        Returns:
            Solution: The constructed and evaluated solution object.

        Raises:
            ValueError: If the constructed solution does not have the expected string length.
        """

        # Build the solution string by selecting the symbol j at position i where the variable is set to 1
        solution_string = [
            j
            for i in range(instance.sequence_length)
            for j in range(instance.alphabet_size)
            if x_vars[i, j].x > 0.5
        ]

        solution = Solution(instance, solution_string)
        if len(solution.sequence) != instance.sequence_length:
            raise ValueError(f'Solution is not complete! It has length {len(solution_string)} rather than {instance.sequence_length}')
        solution.evaluate()
        return solution


    def _build_model(self, instance: Instance, disp:int =0) -> Tuple[gp.Model, dict]:
        """
        Builds and returns a Gurobi optimization model for the 2OMF problem based on the provided instance.
        This method sets up the Gurobi environment and model with specific parameters such as random seed, verbosity, time limit, thread count, MIP gap tolerances, and presolve options. It creates binary decision variables representing the selection of alphabet symbols at each position in a sequence, adds constraints to ensure each position is assigned exactly one symbol, and sets the objective to minimize the sum of squared Hamming distances between the model's solution and the input sequences.
        Args:
            instance (Instance): An object containing the problem data, including the number of sequences, sequence length, alphabet size, and the sequences themselves.
        Returns:
            Tuple[gp.Model, dict]: A tuple containing the constructed Gurobi model and a dictionary of decision variables.
        """

        num_sequences = instance.num_sequences
        sequence_length = instance.sequence_length
        alphabet_size = instance.alphabet_size

        with gp.Env(empty=True) as env:
            env.setParam("Seed", int(self.seed))
            if disp == 0:
                env.setParam("LogToConsole", 0)
                env.setParam("OutputFlag", 0)
            else:
                env.setParam("LogToConsole", 1)
                env.setParam("OutputFlag", 1)
            env.start()
            model = gp.Model(env=env, name="2OMF")
            model.Params.TimeLimit = self.time_limit
            model.Params.Threads = self.threads
            model.Params.MIPGap = self.gap_tolerance
            model.Params.MIPGapAbs = self.gap_tolerance
            model.Params.Presolve = self.presolve
            model.Params.PreQLinearize = self.preqlinearize

            # Create variables
            x = model.addVars(sequence_length, alphabet_size, vtype=GRB.BINARY, name="x")
            model.addConstrs((x.sum(i, '*') == 1 for i in range(sequence_length)), "c0")


            # # Set objective
            objective_terms = []
            for sequence in range(num_sequences):
                hamming_distance_element = gp.quicksum(
                    (1 - x[i, int(instance.sequences[sequence,i])]) for i in range(sequence_length))
                objective_terms.append(hamming_distance_element * hamming_distance_element)
                
            object_expression = gp.quicksum(objective_terms)

            model.setObjective(object_expression, GRB.MINIMIZE)

        return model, x

    def _solve(self, model: gp.Model) -> Tuple[float, float, float]:
        
        try:
            model.optimize()
        except gp.GurobiError as e:
            logger.error(f'Gurobi Error during optimization: {e}')
            raise OptimizationError(f'Gurobi Error during optimization: {e}')

        if self.verbosity_level > 0:
            model.printStats()
        if model.SolCount == 0:            
            logger.warning("No solution found within the time limit.")
        return model.objVal, model.Runtime, model.MIPGap
    

    def _minimize_gurobi(self, instance: Instance, timeout:float, 
              disp:int =0, seed:int =42) -> Result:
        """
        Run the solver on the given instance.

        Parameters
        ----------
        instance : Instance
            The instance to solve.

        Returns
        -------
        result (Result): The optimization result represented as a "Result" object. Important attributes are: ''x'' the solution sequence,
            ''fun'' the value of the objective function at that solution, and ''message'' which describes the cause of the termination.
            See `data/result.py` for more details.
        """
        self.time_limit = timeout
        self.seed = seed
        model, x_vars = self._build_model(instance, disp)

        try: 
            obj, duration, gap = self._solve(model)
            if obj is None:
                return Result(
                    x = None,
                    objective_value=None,
                    duration=model.Runtime,
                    n_iterations=model.IterCount,
                    n_nodes=model.NodeCount,
                    gap = None,
                    lower_bound=model.ObjBound,
                )
            
            solution = self._get_solution(instance, x_vars)
            result = Result(
                x=solution.sequence,
                objective_value=solution.objective_value,
                duration=model.Runtime,
                n_iterations=model.IterCount,
                n_nodes= model.NodeCount,
                gap = model.MIPGap,
                lower_bound = model.ObjBound,
            )
            return result
        finally:
            model.close() # Free all resources associated with this Model object