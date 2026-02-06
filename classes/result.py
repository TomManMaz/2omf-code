
__all__ = ["Result"] 

from _lib._util import _NiceResult




class Result(_NiceResult):
    """
    Represent the optimisation result.

    Attributes:
        duration (float): The total time taken to compute the result, in seconds.
        x (List[int]): The solution represented as a list of integers, typically indices or characters.
        objective_value (float): The objective function value at 'x'.
        message (str): Description of the cause of the termination.
        n_iterations (int): The number of iterations performed during computation.
        n_improvements (Optional[int]): The number of improvements found during the process (if applicable).
        n_nodes (Optional[int]): The number of nodes explored (if applicable).
        gap (Optional[float]): The optimality gap, if applicable.
        lower_bound (Optional[float]): The lower bound of the objective function, if applicable.
    """
    pass
   
