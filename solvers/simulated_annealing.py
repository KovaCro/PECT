"""
Simulated annealing solver module (callable)

Solves PECT problem using simulated annealing
"""

import math
import random
from collections.abc import Callable
import numpy as np
from pect import Pectp, Pects, to_numpy, fast_neighbourhood
from solvers import naive


def solve(
    pect: Pectp,
    temperature: int = 5000,
    temp_update: Callable[[float], float] = lambda x: x * 0.9,
    temp_min: float = 1.0,
) -> Pects:
    """
    Solves PECT problem using simulated annealing

    Args:
        pect: PECT problem
        initial_solution: initial solution (defaults to all events unslotted)
        temperature: initial temperature
        temp_update: temperature update function
        temp_min: termination criteria

    Returns:
        PECT problem
    """

    solution = naive.solve(pect)

    np_pect, np_solution = to_numpy(pect, solution)
    attends = np_pect[5]
    solution_cost = attends[:, np_solution[:, 0] == -1].sum()

    while temperature > temp_min:
        neighbour = np_solution.copy()
        scheduled = np.where(neighbour[:, 0] != -1)[0]
        if scheduled.size != 0:
            k = np.random.choice(scheduled)
            neighbour[k] = [-1, -1]
        inserts = fast_neighbourhood(np_pect, neighbour, (True, False, False), 1.0)
        insert = inserts[np.random.choice(inserts.shape[0])]
        i_e, i_ts, i_r, _, _ = insert
        neighbour[i_e] = [i_ts, i_r]
        n_cost = attends[:, neighbour[:, 0] == -1].sum()

        delta_cost = n_cost - solution_cost

        if delta_cost <= 0:
            np_solution = neighbour
            solution_cost = n_cost
        else:
            if math.e ** (-delta_cost / temperature) > random.uniform(0, 1):
                np_solution = neighbour
                solution_cost = n_cost

        temperature = temp_update(temperature)

    return np_solution.tolist()


def __call__(pect, **kwargs):
    return solve(pect, **kwargs)
