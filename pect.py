""" PECT definition and methods module. """

import os
from utils import consecutive_groups, find_duplicate_indices


# PECT problem
type Pectp = tuple[
    int, int, int, int, list[int], list[int], list[int], list[int], list[int], list[int]
]
# PECT solution
type Pects = list[list[int]]


def parse_problem(path: str) -> Pectp:
    """
    Parses a .tim file. Does not check if file is valid.

    Args:
        path: String to file path

    Returns:
        PECT problem
    """

    with open(path, "r", encoding="utf") as file:
        first_line = file.readline().split(" ")
        m = 45
        n = int(first_line[0])
        r = int(first_line[1])
        f = int(first_line[2])
        s = int(first_line[3])
        room_sizes = []
        attends = []
        roomfeatures = []
        eventfeatures = []
        event_availability = []
        before = []

        for _ in range(r):
            room_sizes.append(int(file.readline()))

        for _ in range(s):
            tmp = []
            for _ in range(n):
                tmp.append(int(file.readline()))
            attends.append(tmp)

        for _ in range(r):
            tmp = []
            for _ in range(f):
                tmp.append(int(file.readline()))
            roomfeatures.append(tmp)

        for _ in range(n):
            tmp = []
            for _ in range(f):
                tmp.append(int(file.readline()))
            eventfeatures.append(tmp)

        for _ in range(n):
            tmp = []
            for _ in range(m):
                tmp.append(int(file.readline()))
            event_availability.append(tmp)

        for _ in range(n):
            tmp = []
            for _ in range(n):
                tmp.append(int(file.readline()))
            before.append(tmp)

        return (
            n,
            r,
            f,
            s,
            room_sizes,
            attends,
            roomfeatures,
            eventfeatures,
            event_availability,
            before,
        )


def parse_solution(path: str, pect: Pectp) -> Pects:
    """
    Parses a .sln file. Does not check if file is valid.

    Args:
        path: String to file path
        pect: Problem for the solution

    Returns:
        PECT solution
    """

    n = pect[0]
    solution = []
    with open(path, "r", encoding="utf") as file:
        for _ in range(n):
            line = file.readline().split(" ")
            solution.append([int(line[0]), int(line[1])])
    return solution


def is_feasible(pect: Pectp, solution: Pects) -> bool:
    """
    Checks if solution is feasible

    Args:
        pect: PECT problem
        solution: PECT solution

    Returns:
        True if solution is feasible, False otherwise
    """

    for s in solution:
        if s[0] == -1:
            return False
    return is_valid(pect, solution)


def is_valid(pect: Pectp, solution: Pects, fails: list[object] = None) -> bool:
    """
    Checks if solution is valid

    Args:
        pect: PECT problem
        solution: PECT solution
        fails: Optional list to which validation fails are appended (should be empty).

    Returns:
        True if solution is valid, False otherwise
    """

    (
        n,
        _,
        _,
        s,
        room_sizes,
        attends,
        roomfeatures,
        eventfeatures,
        event_availability,
        before,
    ) = pect

    # IMPORTANT:
    # When event fails, it gets ignored in future checks
    # uninserted events are ignored by default
    ignored = set()
    for i in range(n):
        if solution[i][0] == -1 or solution[i][1] == -1:
            ignored.add(i)

    # 1. no student attends more than one event at the same time
    # Only first event is considered valid
    # if there is more than one attended at the same time
    for student in range(s):
        attended_timeslots = [
            solution[event][0]
            for event in range(n)
            if attends[student][event] and event not in ignored
        ]
        unattendable_events = find_duplicate_indices(attended_timeslots)
        if unattendable_events:
            if fails is not None:
                ignored.update(unattendable_events)
                fails.extend(
                    [{"rule": 1, "event": event} for event in unattendable_events]
                )
            else:
                return False
    # 2. the room is big enough for all the attending students
    # and satisfies all the features required by the event
    for event in range(n):
        if event in ignored:
            continue
        room = solution[event][1]
        attendees = 0
        for student in range(s):
            attendees += attends[student][event]
        if room_sizes[room] < attendees:
            if fails is not None:
                ignored.add(event)
                fails.append({"rule": 2, "event": event})
            else:
                return False
        if not all(
            not req or (req and has)
            for req, has in zip(eventfeatures[event], roomfeatures[room])
        ):
            if fails is not None:
                ignored.add(event)
                fails.append({"rule": 2, "event": event})
            else:
                return False
    # 3. only one event is put into each room in any timeslot
    # Only first event is considered valid
    # if there is more than one event occupying same room at the same time
    occupied = {}
    for event in range(n):
        if event in ignored:
            continue
        timeslot = solution[event][0]
        room = solution[event][1]
        if room in occupied:
            if timeslot in occupied[room]:
                if fails is not None:
                    ignored.add(event)
                    fails.append({"rule": 3, "event": event})
                else:
                    return False
            else:
                occupied[room].append(timeslot)
        else:
            occupied[room] = [timeslot]
    del occupied
    # 4. events are only assigned to timeslots
    # that are pre-defined as available for those events
    for event in range(n):
        if event in ignored:
            continue
        timeslot = solution[event][0]
        if event_availability[event][timeslot] != 1:
            if fails is not None:
                ignored.add(event)
                fails.append({"rule": 4, "event": event})
            else:
                return False
    # 5. where specified, events are scheduled to occur in the correct order in the week
    # If fail occurs, second event in order is considered invalid
    for i in range(n):
        for j in range(i, n):
            if i in ignored or j in ignored:
                continue
            if before[i][j] == -1 and solution[j][0] >= solution[i][0]:
                if fails is not None:
                    ignored.add(j)
                    fails.append({"rule": 5, "event": j})
                else:
                    return False

    if fails:
        return False

    return True


def make_valid(pect: Pectp, solution: Pects) -> Pects:
    """
    Creates a valid solution from the invalid

    Args:
        pect: PECT problem
        solution: PECT solution

    Returns:
        Valid solution if given solution is invalid, otherwise given solution
    """

    validation_fails = []
    if not is_valid(pect, solution, validation_fails):
        invalid_events = [fail["event"] for fail in validation_fails]
        for invalid_event in invalid_events:
            solution[invalid_event][0] = -1
            solution[invalid_event][1] = -1
    return solution


def evaluate(pect: Pectp, solution: Pects) -> tuple[int, int]:
    """
    Evaluates PECT solution

    Args:
        pect: PECT problem
        solution: PECT solution

    Returns:
        Distance to feasibility and soft cost
    """

    (
        n,
        _,
        _,
        s,
        _,
        attends,
        _,
        _,
        _,
        _,
    ) = pect

    distance_to_feasibility = 0
    soft_cost = 0
    # feasibility distance
    for event in range(n):
        if solution[event][0] == -1:
            for student in range(s):
                distance_to_feasibility += attends[student][event]
    # soft penals
    for student in range(s):
        attended_timeslots = [
            solution[event][0]
            for event in range(n)
            if attends[student][event] and solution[event] != -1
        ]
        days = []
        for i in range(5):
            days.append(
                [
                    timeslot % 9
                    for timeslot in attended_timeslots
                    if i * 9 <= timeslot < (i + 1) * 9
                ]
            )
        # a student has a class in the last slot of the day
        soft_cost += sum(list(map(lambda x: x.count(8), days)))
        # a student has more than two classes consecutively
        for day in days:
            consecutive_classes = consecutive_groups(sorted(day))
            soft_cost += sum(
                len(group) - 2 for group in consecutive_classes if len(group) > 2
            )
        # a student has a single class on a day
        for day in days:
            if len(day) == 1:
                soft_cost += 1

    return distance_to_feasibility, soft_cost


def write_solution(path: str, pect: Pectp, solution: Pects) -> None:
    """
    Writes a formatted solution to file

    Args:
        path: string to file path
        pect: PECT problem
        solution: PECT solution
    """

    (
        _,
        r,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = pect

    timetable = [[["-" for _ in range(5)] for _ in range(9)] for _ in range(r)]
    for i, event in solution:
        timeslot, room = event
        slot = timeslot % 9
        day = timeslot // 9
        timetable[room][slot][day] = str(i)
    with open(path, "w", encoding="utf-8") as file:
        for room in range(r):
            file.write(f"ucionica{room}\n")
            file.write("        PON     UTO     SRI     CET     PET\n")
            for slot in range(9):
                file.write("0       ")
                for day in range(5):
                    file.write(timetable[room][slot][day].ljust(8))
                file.write("\n")
            file.write("\n")


def write_evaluation(path: str, evaluation: tuple[int, int]) -> None:
    """
    Writes a formatted evaluation to file

    Args:
        path: string to file path
        evaluation: tuple representing distance to feasibility and soft cost
    """

    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Instance: {os.path.split(path)[1]}\n")
        file.write(f"Distance to feasibility: {evaluation[0]}\n")
        file.write(f"Soft cost: {evaluation[1]}\n")
        file.write(f"Total: {evaluation[0] + evaluation[1]}\n")
        file.write("\n")
