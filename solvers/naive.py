""" 
Naive solver module (callable)

Solves PECT problem using naive approach (greedy without optimal greedy step)

Args:
    pect: PECT problem

Returns:
    PECT problem
"""

import sys
from pect import Pectp, Pects


def solve(pect: Pectp) -> Pects:
    """
    Solves PECT problem using naive approach

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

    solution = [[-1, -1] for _ in range(n)]
    room_satisfies_event = [[1 for _ in range(n)] for _ in range(r)]
    attends_transposed = [
        [attends[student][event] for student in range(s)] for event in range(n)
    ]
    attendees = [sum(attends_transposed[event]) for event in range(n)]

    available_events = set(range(n))

    for room in range(r):
        for event in range(n):
            if attendees[event] > room_sizes[room]:
                room_satisfies_event[room][event] = 0
                continue
            if not all(
                not req or (req and has)
                for req, has in zip(eventfeatures[event], roomfeatures[room])
            ):
                room_satisfies_event[room][event] = 0
                continue

    for timeslot in range(45):
        busy_students = [0 for _ in range(s)]
        timeslot_unavailable_events = set()
        for room in range(r):
            to_be_removed = set()
            for event in available_events:
                if event in timeslot_unavailable_events:
                    continue
                if not room_satisfies_event[room][event]:
                    continue
                if any(
                    s1 and s2
                    for s1, s2 in zip(busy_students, attends_transposed[event])
                ):
                    continue
                if not event_availability[event][timeslot]:
                    timeslot_unavailable_events.add(event)
                    continue
                solution[event][0] = timeslot
                solution[event][1] = room
                busy_students = [
                    busy_students[student] or attends[student][event]
                    for student in range(s)
                ]
                for i in range(event + 1, n):
                    if before[event][i] == -1:
                        if i in available_events:
                            to_be_removed.add(i)
                for available_event in available_events:
                    if before[event][available_event] == 1:
                        timeslot_unavailable_events.add(available_event)
                to_be_removed.add(event)
                break
            available_events = available_events - to_be_removed

    return solution


solve.__name__ = "naive"
sys.modules[__name__] = solve
