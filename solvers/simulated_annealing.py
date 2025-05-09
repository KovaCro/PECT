""" 
Simulated annealing solver module (callable)

Solves PECT problem using simulated annealing

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
    Solves PECT problem using simulated annealing

    Args:
        pect: PECT problem

    Returns:
        PECT problem
    """

    (
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
    ) = pect

    # pylint: disable=not-callable
    solution = naive(pect)
    # TODO

    return solution

solve.__name__ = 'simulated_annealing'
sys.modules[__name__] = solve
