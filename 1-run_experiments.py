# Standard library
import time
from pathlib import Path
import argparse
import sys

# Local application
from algorithms.minimize import minimize, METHODS
from classes.instance import Instance
from utils.gracefull_killer import GracefulKiller
from utils.logging import get_logger
from utils.logging import configure_logging                                                                                                                                                 
configure_logging()      



def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the program.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        --instance_file, -i (str, required): Path to the instance file.
        --seed, -s (int, optional): Random seed (default: 42).
        --algorithm, -a (str, required): Algorithm to run. Must be one of ALGORITHMS.
        --timeout (int, optional): Timeout in seconds (default: 3600).
        --initial_temperature (float, optional): Initial temperature for Simulated Annealing (default: 3643.2046).
        --final_temperature (float, optional): Final temperature for Simulated Annealing (default: 0.4053).
        --verbose, -v (flag, optional): Enable verbose mode.
        --output_dir (Path, optional): Output directory for results (default: ~/2omf/experiments).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_file', '-i', required=True, type=str, help='instance file')
    parser.add_argument('--seed', '-s', required=False, type=int, default=42, help='random seed')
    parser.add_argument('--algorithm', '-a', required=True, type=str, choices=METHODS, help='algorithm to run ')
    parser.add_argument('--timeout', required=False, type=int,default=10, help="Timeout in seconds")
    # hg2 parameters                                                                                                                 
    parser.add_argument('--population_size', type=int, default=None,                                                                                                                        
        help='HG2 population size (default: 81 from config)')                                                                                                                   
    parser.add_argument('--mutation_rate', type=float, default=None,                                                                                                                        
        help='HG2 mutation rate (default: 0.3043 from config)')   
    parser.add_argument('--num_offspring', type=int, default=None,                                                                                                                        
        help='HG2 number of offspring (default: 160 from config)')                                                                                                                   
    parser.add_argument('--num_paired_parents', type=int, default=None,                                                                                                                        
        help='HG2 number of paired parents (default: 55 from config)')                                                                                                                   
    parser.add_argument('--num_elites', type=int, default=None,                                                                                                                        
        help='HG2 number of elites (default: 4 from config)')       
    # simulated annealing parameters
    parser.add_argument('--initial_temperature', type=float, default=None, help='SA initial temperature')
    parser.add_argument('--final_temperature', type=float, default=None, help='SA final temperature')
    parser.add_argument('--verbose', '-v', required=False, action='store_true', help='verbose mode')
    parser.add_argument('--output_dir', type=Path, default=Path.home() / '2omf' / 'experiments', help='output directory for results')
    return parser.parse_args()



def main():
    start_time = time.perf_counter()
    logger = get_logger(__name__)
    logger.info(f'Using {sys.version}')
    args = parse_arguments()
    logger.info(f'Arguments: {args}')
    method = args.algorithm
    instance = Instance.from_file(args.instance_file)
    logger.info(f'Loaded instances {instance.name} with {instance.num_sequences} strings of length {instance.sequence_length}')


    options = {}
    if args.verbose:
        options['disp'] = True
    if args.seed:
        seed = args.seed
        if seed > 2**31:
            seed = args.seed % 2**31
            logger.warning(f'Reducing random seed from {args.seed} to {seed}')
        options['seed'] = seed
    if args.algorithm in ['simulated_annealing', 'sa']:
        options['initial_temperature'] = args.initial_temperature                                                                                                                                   
        options['final_temperature'] = args.final_temperature
 
    if args.population_size is not None:                                                                                                                                                    
      options['population_size'] = args.population_size     
    if args.mutation_rate is not None:                                                                                                                                                    
      options['mutation_rate'] = args.mutation_rate     
    if args.num_offspring is not None:                                                                                                                                                    
      options['num_offspring'] = args.num_offspring     
    if args.num_paired_parents is not None:                                                                                                                                                    
      options['num_paired_parents'] = args.num_paired_parents     
    if args.num_elites is not None:                                                                                                                                                    
      options['num_elites'] = args.num_elites

    

    result = minimize(instance=instance,
                      method=method,
                      timeout=args.timeout, options=options)


    killer = GracefulKiller()  # Handles SIGINT and SIGTERM
    



    experiment_path = Path.home() / '2omf' / 'experiments' / 'summary'
    experiment_path.mkdir(parents=True, exist_ok=True)
    # while not killer.kill_now:
   
    if killer.kill_now:
        logger.info("Received termination signal. Stopping experiment gracefully.")
    # test result.__dir__

    result.log_result(args.algorithm)
    summary_file = Path(experiment_path  / f'{instance.name}_{args.algorithm}_{args.seed}.csv')
    result.to_file(args.seed, instance, args.algorithm, summary_file)
    logger.info(f'Summary written to {summary_file}')

    logger.info(f'Completely finished after {time.perf_counter() - start_time:.2f}s')



    # runner = ExperimentRunner(args)
    # runner.run()

if __name__ == '__main__':
    logger = get_logger(__name__)
    main()