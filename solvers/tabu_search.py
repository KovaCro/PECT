""" 
Tabu search solver module (callable)

Solves PECT problem using tabu search

Args:
    pect: PECT problem

Returns:
    PECT problem
"""

import sys
from pect import Pectp, Pects
from solvers import naive


def solve(pect: Pectp) -> Pects:
    """
    Solves PECT problem using tabu search

    Args:
        pect: PECT problem

    Returns:
        PECT problem
    """

    [
        n,
        r,
        _,
        s,
        room_sizes,
        attends,
        roomfeatures,
        eventfeatures,
        event_availability,
        before,
    ] = pect

    solution = naive(pect)
    # TODO

    return solution

solve.__name__ = 'tabu_search'
sys.modules[__name__] = solve
