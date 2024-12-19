from pathlib import Path
import inspect
from collections.abc import Callable
from solvers import naive
import pect


def generate_and_evaluate_solutions(
    input_paths: list[Path],
    output_dir: Path,
    solver: Callable[[pect.Pectp], pect.Pects],
    progress: bool = True,
) -> None:
    """
    Uses solver to generate solutions and writes solutions
    to a directory toggether with solution evaluations.

    Args:
        input_paths: list of paths to problems
        output_dir: path to output directory
        solver: function used to generate solution
        progress: whether to print progress or not (optional)
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    solver_type = inspect.getmodule(solver).__name__

    for i, input_path in enumerate(input_paths):
        if progress:
            print(f"Generating {solver_type} solutions: {i+1}/{len(input_paths)}")
        problem = pect.parse_problem(input_path)
        solution = solver(problem)
        is_valid = pect.is_valid(problem, solution)
        evaluation = pect.evaluate(problem, solution)
        output_path = output_dir / input_path.stem
        pect.write_formatted_solution(output_path.with_suffix(".formatted.sln"), problem, solution)
        pect.write_solution(output_path.with_suffix(".sln"), solution)
        pect.write_evaluation(output_path.with_suffix(".eval"), evaluation)
        if progress:
            print(f"Generated {solver_type} solution: {i+1}/{len(input_paths)}, Valid: {is_valid}, Cost: {evaluation[0] + evaluation[1]}")


if __name__ == "__main__":
    problem_paths = list(Path("./datasets").iterdir())[:3]
    #generate_and_evaluate_solutions(
    #    problem_paths, Path("./solutions/naive"), naive
    #)
    #generate_and_evaluate_solutions(
    #    problem_paths, Path("./solutions/greedy"), greedy
    #)
    #generate_and_evaluate_solutions(
    #    problem_paths, Path("./solutions/tabu"), tabu_search
    #)
