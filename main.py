from solve import generate_and_evaluate_solutions
from pathlib import Path
from solvers import *

if __name__ == "__main__":
    #problem_paths = list(Path(sys.argv[1]).iterdir())
    #solvers = sys.argv[2].split('|')
    problem_paths = list(Path('./datasets').iterdir())
    #generate_and_evaluate_solutions(
    #   problem_paths, Path("./solutions/naive"), naive
    #)
    #generate_and_evaluate_solutions(
    #    problem_paths, Path("./solutions/greedy"), greedy
    #)
    generate_and_evaluate_solutions(
        problem_paths, Path("./solutions/hill_climbing"), hill_climbing
    )