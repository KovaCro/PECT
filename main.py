from solve import generate_and_evaluate_solutions
from pathlib import Path
from solvers import *

if __name__ == "__main__":
    # problem_paths = list(Path(sys.argv[1]).iterdir())
    # solvers = sys.argv[2].split('|')
    problem_paths = list(Path("./datasets").iterdir())

    
    # Example call
    # generate_and_evaluate_solutions(
    #     problem_paths,
    #     Path("./solutions/hill_climbing_best_type1"),
    #     hill_climbing.solve,
    #     {"num_iter": 1000, "tactic": "best", "neighbourhood_type": 1},
    # )
    generate_and_evaluate_solutions(
        problem_paths,
        Path("./solutions/simulated_annealing_default"),
        simulated_annealing.solve,
        #{"aspir": True},
    )
