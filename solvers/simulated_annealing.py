""" 
Simulated annealing solver module (callable)

Solves PECT problem using simulated annealing

Args:
    pect: PECT problem

Returns:
    PECT problem
"""

from pect import Pectp, Pects
from solvers import naive


def solve(pect: Pectp) -> Pects:
    """
    Solves PECT problem using simulated annealing

    Args:
        pect: PECT problem
        num_iter: Number of iterations

    Returns:
        PECT problem
    """

    # TODO

    #return solution


def __call__(pect, **kwargs):
    return solve(pect, **kwargs)
