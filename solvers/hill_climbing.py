"""
Hill climbing solver module (callable)

Solves PECT problem using hill climbing

Args:
    pect: PECT problem

Returns:
    PECT problem
"""

import sys
import numpy as np
from typing import Literal
from tqdm import tqdm
from pect import Pectp, Pects, to_numpy, vectorized_fast_neighbourhood
from solvers import naive


def solve(
    pect: Pectp,
    num_iter: int = 1000,
    tactic: Literal["best", "first"] = "best",
    neighbourhood_moves: tuple[bool, bool, bool] = (True, True, True),
) -> Pects:
    """
    Solves PECT problem using hill climbing

    Args:
        pect: PECT problem
        num_iter: Number of iterations
        neighbourhood_moves: tuple of booleans (insert, extract, swap) indicating which types of moves to include in the neighbourhood

    Returns:
        PECT problem
    """

    # pylint: disable=not-callable
    solution = naive(pect)

    np_pect, np_solution = to_numpy(pect, solution)

    for _ in tqdm(range(num_iter)):
        neighbourhood = vectorized_fast_neighbourhood(np_pect, np_solution, neighbourhood_moves)
        neighbour_ind = None

        if not neighbourhood.size:
            break

        # Note: does not consider soft cost

        if tactic == "best":
            neighbour_ind = np.argmin(neighbourhood[:, 3])

        if tactic == "first":
            neighbour_ind = np.where(neighbourhood[:, 3] < 0)[0]

        neighbour = neighbourhood[neighbour_ind]

        if neighbour[0] == -1:
            tmp = np_solution[neighbour[1]]
            np_solution[neighbour[1]] = np_solution[neighbour[2]]
            np_solution[neighbour[2]] = tmp
        else:
            np_solution[neighbour[0]][0] = neighbour[1]
            np_solution[neighbour[0]][1] = neighbour[2]

    return solution


solve.__name__ = "hill_climbing"
sys.modules[__name__] = solve
