"""
Unified interfaces to minimization algorithms.
"""
from typing import Optional, Dict

from classes.instance import Instance
from classes.result import Result
from utils.logging import get_logger                                                                                                                                                              
logger = get_logger(__name__)   







METHODS = ['simulated_annealing', 'sa', 'hg2', 'gurobi']

def minimize(instance: Instance, 
          method: str, 
          timeout: float=60.0,
          options: Optional[Dict] = None) -> Result:
    """
    Minimize the given instance using the specified method.
    Args:
      instance (Instance): The instance to be minimized.
      method: algorithm.  Should be one of

        - 'simulated_annealing' :ref:`(see here)                <algorithms._simulated_annealing>`
        - 'greedy' :ref:`(see here)                             <algorithms.greedy>`
        - 'gain_heuristic' :ref:`(see here)                     <algorithms.gain_heuristic>`
        - 'hg2' :ref:`(see here)                                <algorithms.hg2>`
        - 'gurobi' :ref:`(see here)                             <algorithms.exact>`
      timeout (float): Timeout in seconds for the algorithm to run. Default is 60 seconds.
      options (dict, optional): Additional options for the algorithm, if applicable.
        disp : bool
          Set to true to print messages
  
      
    Returns:
        res : Result
        The optimization result represented as a ``Result`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `Result` for a description of other attributes.
        """
    if method not in METHODS:
      raise ValueError(f"Method '{method}' is not supported. Choose from {METHODS}.")


    if options is None:
        options = {}
    logger.info(f'Starting minimization with method {method} on instance {instance.name} with timeout {timeout} seconds.')
    logger.info(f'Options: {options}  ')

    if method == 'simulated_annealing' or method == 'sa':     
      from algorithms.simulated_annealing import SimulatedAnnealing
      initial_temp = options.pop('initial_temperature', None)
      final_temp = options.pop('final_temperature', None)
      result = SimulatedAnnealing(initial_temperature=initial_temp,
                                      final_temperature=final_temp)._minimize_sa(instance, timeout=timeout, **options)
    elif method == 'hg2':
      from algorithms.hg2 import HG2
      result = HG2().minimize_hg2(instance, timeout=timeout, **options)
    elif method == 'gurobi':
      from algorithms.gurobi import GurobiWrapper, GUROBI_AVAILABLE
      if not GUROBI_AVAILABLE:
          raise ImportError("Gurobi is not installed. " \
          "Please install Gurobi to use this method." \
          "Available algorithms are hg2, Simulated Annealing, and gain_heuristic.")
      result = GurobiWrapper()._minimize_gurobi(instance=instance, timeout=timeout, **options)

    else:
      raise ValueError(f'Unknown solver {method}')

    return result
