import argparse
from collections.abc import Callable
import json
from pathlib import Path
from solve import generate_and_evaluate_solutions
from solvers import hill_climbing, greedy, naive, tabu_search
from pect import Pectp, Pects


def get_accessible_solver(name: str) -> Callable[[Pectp], Pects]:
    """Helper getter function for solvers."""
    if name == "naive":
        return naive.solve
    if name == "greedy":
        return greedy.solve
    if name == "hill_climbing":
        return hill_climbing.solve
    if name == "tabu_search":
        return tabu_search.solve


def main():
    """
    This script parses command-line arguments, generates and evaluates 
    solutions for problems using the chosen optimization algorithm.

    Command-line arguments:
        -input <path>: Required. Path to the dataset directory containing problem files.
                    E.g., A:\\PECT\\dataset\\
        -output <path>: Required. Path to the directory where solutions will be saved.
                        E.g., A:\\PECT\\solutions\\
        -workers <int>: Optional. Number of worker processes to use for computation.
                        If not specified, the number of CPUs will be used.
        -solver <str>: Required. The optimization algorithm to use.
                    Choices: "naive", "greedy", "hill_climbing", "tabu_search".
        -args <str>: Optional. JSON string of additional arguments to pass to the
                    selected solver. E.g., '{"num_iter": 1000}'
    """
    parser = argparse.ArgumentParser(
        description="PECT problem solver.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-input",
        type=str,
        required=True,
        help="Path to the dataset directory.\nExample: A:\\PECT\\dataset\\",
    )
    parser.add_argument(
        "-output",
        type=str,
        required=True,
        help="Path to the output directory.\nExample: A:\\PECT\\solutions\\",
    )
    parser.add_argument(
        "-workers",
        type=int,
        default=None,
        help="Number of worker processes to use for computation.",
    )
    parser.add_argument(
        "-solver",
        type=str,
        required=True,
        choices=["naive", "greedy", "hill_climbing", "tabu_search"],
        help="The optimization algorithm to use.\nAvailable choices: naive, greedy, hill_climbing, tabu_search",
    )
    parser.add_argument(
        "-args",
        type=str,
        default="{}",
        help="Optional arguments for the solver in JSON format.\nExample: '{\"num_iter\": 1000}'",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_dir():
        print(
            f"Warning: Input path '{args.input}' is not a valid directory or does not exist."
        )
    else:
        print(f"Input dataset path: {input_path.resolve()}")

    if not output_path.is_dir():
        print(
            f"Warning: Output path '{args.input}' is not a valid directory or does not exist."
        )
    else:
        print(f"Output path: {input_path.resolve()}")

    if args.workers is not None:
        print(f"Number of workers: {args.workers}")
    else:
        print("Number of workers will be set to number of CPUs.")

    print(f"Solver chosen: {args.solver}")

    solver_args = {}
    try:
        solver_args = json.loads(args.args)
        print(f"Solver arguments: {solver_args}")
    except json.JSONDecodeError:
        print(f"Error: Could not parse -args as valid JSON: '{args.args}'")
    except Exception as e:
        print(f"An unexpected error occurred while processing solver arguments: {e}")

    print(solver_args)
    problem_paths = list(input_path.iterdir())
    generate_and_evaluate_solutions(
        problem_paths,
        output_dir=output_path,
        solver=get_accessible_solver(args.solver),
        max_workers=args.workers,
        solver_kwargs=solver_args,
    )
    print("Program finished.")


if __name__ == "__main__":
    main()
