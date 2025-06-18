"""
Hill climbing solver module (callable)

Solves PECT problem using hill climbing

Args:
    pect: PECT problem

Returns:
    PECT problem
"""

from typing import Literal
import numpy as np
from tqdm import tqdm
from pect import Pectp, Pects, to_numpy, fast_neighbourhood
from solvers import naive


def solve(
    pect: Pectp,
    num_iter: int = 1000,
    tactic: Literal["best", "first"] = "best",
    neighbourhood_type: Literal[1, 2] = 1,
) -> Pects:
    """
    Solves PECT problem using hill climbing

    Args:
        pect: PECT problem
        num_iter: number of iterations
        tactic: 'best' chooses best neighbour each iteration, 'first' chooses first improving neighbour
        neighbourhood_type: 1 for insert/extract, 2 for swap between slotted

    Returns:
        PECT problem
    """

    solution = naive.solve(pect)

    np_pect, np_solution = to_numpy(pect, solution)

    # TODO: neighbourhood type 2 should be implemented differently

    for _ in tqdm(range(num_iter)):
        neighbourhood = None
        neighbour_ind = None

        if neighbourhood_type == 1:
            neighbourhood = fast_neighbourhood(
                np_pect, np_solution, (False, False, True), 0.2
            )

        if neighbourhood_type == 2:
            scheduled_events = np.where(np_solution[:, 0] != -1)[0]
            if scheduled_events.size != 0:
                rand_ind = np.random.choice(scheduled_events)
                np_solution[rand_ind][0] = -1
                np_solution[rand_ind][1] = -1
            neighbourhood = fast_neighbourhood(
                np_pect, np_solution, (True, False, False), 0.2
            )

        if not neighbourhood.size:
            continue

        if tactic == "best":
            neighbour_ind = np.lexsort((neighbourhood[:, 4], neighbourhood[:, 3]))[0]

        if tactic == "first":
            neighbour_ind = np.where(
                (neighbourhood[:, 3] < 0) | ((neighbourhood[:, 3] == 0) & (neighbourhood[:, 4] < 0))
            )[0]

            if neighbour_ind.size == 0:
                continue

            neighbour_ind = neighbour_ind[0]

        neighbour = neighbourhood[neighbour_ind]
        if neighbour[0] == -1:
            tmp = np_solution[neighbour[1]].copy()
            np_solution[neighbour[1]] = np_solution[neighbour[2]].copy()
            np_solution[neighbour[2]] = tmp
        else:
            np_solution[neighbour[0]][0] = neighbour[1]
            np_solution[neighbour[0]][1] = neighbour[2]

    return np_solution.tolist()

def __call__(pect, **kwargs):
    return solve(pect, **kwargs)
