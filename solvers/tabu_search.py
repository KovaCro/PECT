"""
Tabu search solver module (callable)

Solves PECT problem using tabu search
"""

import numpy as np
from pect import Pectp, Pects, to_numpy, fast_neighbourhood, evaluate


def solve(
    pect: Pectp,
    initial_solution: Pects | None = None,
    num_iter: int = 1000,
    neighbourhood_sampling_rate: float = 0.2,
    tabu_size: int = 10,
    aspir: bool = False,
) -> Pects:
    """
    Solves PECT problem using tabu search

    Args:
        pect: PECT problem
        initial_solution: initial solution (defaults to all events unslotted)
        num_iter: number of iterations
        neighbourhood_sampling_rate: float indicating the fraction of moves to be processed
        tabu_size: size of the tabu list
        aspir: boolean indicating whether to use aspiration or not

    Returns:
        PECT problem
    """

    n = pect[0]

    if initial_solution is None:
        initial_solution = [[-1, -1] for _ in range(n)]

    np_pect, np_solution = to_numpy(pect, initial_solution)
    tabu_mask = np.zeros((n), dtype=np.bool_)
    tabu_list = -1 * np.ones((tabu_size), dtype=np.int32)
    tabu_ind = 0

    def update_tabu(new_tabu):
        nonlocal tabu_ind
        if tabu_ind == tabu_size:
            tabu_ind = 0
        if tabu_list[tabu_ind] == -1:
            tabu_list[tabu_ind] = new_tabu
            tabu_mask[new_tabu] = True
            tabu_ind += 1
        else:
            curr_tabu = tabu_list[tabu_ind]
            tabu_mask[curr_tabu] = False
            tabu_list[tabu_ind] = new_tabu
            tabu_mask[new_tabu] = True
            tabu_ind += 1

    best_cost = np.array([float("inf"), float("inf")])

    for _ in range(num_iter):
        neighbourhood = fast_neighbourhood(
            np_pect, np_solution, sampling_rate=neighbourhood_sampling_rate
        )

        if not neighbourhood.size:
            continue

        tabu_conflict = (
            (neighbourhood[:, 0] == -1)
            & (tabu_mask[neighbourhood[:, 1]] | tabu_mask[neighbourhood[:, 2]])
        ) | (
            (neighbourhood[:, 0] != -1) & tabu_mask[neighbourhood[:, 0]]
        )  # (k,)

        if aspir:
            current_cost = np.array(evaluate(pect, np_solution.tolist()))
            if (current_cost[0] < best_cost[0]) or (
                (current_cost[0] == best_cost[0]) and (current_cost[1] < best_cost[1])
            ):
                best_cost = current_cost
            new_costs = current_cost + neighbourhood[:, 3:5]
            aspiration_moves = tabu_conflict & (
                (new_costs[:, 0] < best_cost[0])
                | ((new_costs[:, 0] == best_cost[0]) & (new_costs[:, 0] < best_cost[1]))
            )
            tabu_conflict = tabu_conflict & (~aspiration_moves)

        neighbourhood = neighbourhood[~tabu_conflict]  # (k',)

        if not neighbourhood.size:
            continue

        neighbour_ind = np.lexsort((neighbourhood[:, 4], neighbourhood[:, 3]))[0]

        neighbour = neighbourhood[neighbour_ind]
        if neighbour[0] == -1:
            e1, e2 = neighbour[1], neighbour[2]
            tmp = np_solution[e1].copy()
            np_solution[e1] = np_solution[e2].copy()
            np_solution[e2] = tmp
            update_tabu(e1)
            update_tabu(e2)
        else:
            e, ts, r = neighbour[:3]
            np_solution[e][0] = ts
            np_solution[e][1] = r
            update_tabu(e)

    return np_solution.tolist()


def __call__(pect, **kwargs):
    return solve(pect, **kwargs)
