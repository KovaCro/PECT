"""PECT definition and methods module."""

from pathlib import Path
import numpy as np
from utils import consecutive_groups, find_duplicate_indices

# PECT problem
type Pectp = tuple[
    int, int, int, int, list[int], list[int], list[int], list[int], list[int], list[int]
]
# PECT solution
type Pects = list[list[int]]


def parse_problem(path: Path) -> Pectp:
    """
    Parses a .tim file. Does not check if file is valid.

    Args:
        path: String to file path

    Returns:
        PECT problem
    """

    with open(path, "r", encoding="utf") as file:
        first_line = file.readline().split(" ")
        t = 45
        n = int(first_line[0])
        r = int(first_line[1])
        f = int(first_line[2])
        s = int(first_line[3])
        room_sizes = [int(file.readline()) for _ in range(r)]
        attends = [[int(file.readline()) for _ in range(n)] for _ in range(s)]
        roomfeatures = [[int(file.readline()) for _ in range(f)] for _ in range(r)]
        eventfeatures = [[int(file.readline()) for _ in range(f)] for _ in range(n)]
        event_availability = [
            [int(file.readline()) for _ in range(t)] for _ in range(n)
        ]
        before = [[int(file.readline()) for _ in range(n)] for _ in range(n)]

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


def parse_solution(path: Path, pect: Pectp) -> Pects:
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


def to_numpy(pect: Pectp, solution: Pects) -> tuple:
    """
    Turns pect problem and solution lists to numpy arrays

    Args:
        pect: PECT problem
        solution: PECT solution

    Returns:
        Numpy arrays of problem and solution
    """
    (
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
    ) = pect

    return (
        (
            n,
            r,
            f,
            s,
            np.array(room_sizes, dtype=np.uint32),
            np.array(attends, dtype=np.bool_),
            np.array(roomfeatures, dtype=np.bool_),
            np.array(eventfeatures, dtype=np.bool_),
            np.array(event_availability, dtype=np.bool_),
            (np.array(before) == 1).astype(np.bool_),
        ),
        np.array(solution, dtype=np.int32),
    )


def fast_neighbourhood(
    np_pect: any,
    np_solution: any,
    moves: tuple[bool, bool, bool] = (True, True, True),
) -> np.ndarray:
    """
    Efficiently generates and returns neighbourhood of a solution.
    In case the neighbour is generated by swap step, array is in the form
    [event, timeslot, room, relative distance to feasibility, relative soft cost].
    In case the neighbour is generated by swap step, array is in the form
    [-1, event1, event2, relative distance to feasibility, relative soft cost]

    Args:
        np_pect: numpy PECT problem
        np_solution: numpy PECT solution
        moves: tuple of booleans (insert, extract, swap) indicating which types of moves to include in the neighbourhood

    Returns:
        Array of neighbours
    """
    (
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
    ) = np_pect

    t = 45
    T = 9
    solution = np_solution
    neighbours = []

    # how many students per event
    attendees = np.sum(attends, axis=0, dtype=np.int32)

    # is room valid for event
    room_satisfies_event = (room_sizes[:, None] >= attendees[None, :]) & np.all(
        (~eventfeatures[None, :, :] | roomfeatures[:, None, :]), axis=2
    )

    unscheduled_events = np.where(solution[:, 0] == -1)[0]
    scheduled_events = np.where(solution[:, 0] != -1)[0]

    occupied_pairs = set(tuple(pair) for pair in solution[scheduled_events])

    ts_students_bool = np.zeros((t, s), dtype=np.bool_)
    for e in scheduled_events:
        ts = solution[e, 0]
        ts_students_bool[ts] |= attends[:, e]

    def compute_delta_consec_insert(d, h, attending):
        A = ts_students_bool[d * T : (d + 1) * T, attending]
        k_left = (
            np.sum(np.cumprod(A[h - 1 :: -1, :], axis=0), axis=0)
            if h > 0
            else np.zeros(sum(attending), dtype=int)
        )
        k_right = (
            np.sum(np.cumprod(A[h + 1 :, :], axis=0), axis=0)
            if h < T - 1
            else np.zeros(sum(attending), dtype=int)
        )
        delta_per_student = (
            -np.maximum(k_left - 2, 0)
            - np.maximum(k_right - 2, 0)
            + np.maximum(k_left + k_right + 1 - 2, 0)
        )
        return np.sum(delta_per_student)

    def compute_delta_consec_extract(d, h, attending):
        A = ts_students_bool[d * T : (d + 1) * T, attending]
        L_left = np.sum(np.cumprod(A[h::-1, :], axis=0), axis=0)
        L_right = np.sum(np.cumprod(A[h:, :], axis=0), axis=0)
        k = L_left + L_right - 1
        delta_per_student = (
            -np.maximum(k - 2, 0)
            + np.maximum(L_left - 1 - 2, 0)
            + np.maximum(L_right - 1 - 2, 0)
        )
        return np.sum(delta_per_student)

    # Insert, Extract, Swap

    # Insert:
    if moves[0]:
        for event in unscheduled_events:
            attending = attends[:, event]

            # Condition 4
            valid_timeslots = np.where(event_availability[event])[0]
            # Condition 2
            valid_rooms = np.where(room_satisfies_event[:, event])[0]

            for ts in valid_timeslots:
                for room in valid_rooms:
                    # Condition 3
                    if (ts, room) in occupied_pairs:
                        continue

                    # Condition 5
                    successors = np.where(before[event])[0]
                    if np.any(
                        (solution[successors, 0] != -1) & (solution[successors, 0] < ts)
                    ):
                        continue

                    # Condition 1
                    if np.any(attending & ts_students_bool[ts]):
                        continue

                    day = ts // T
                    hour = ts % T
                    num_events_day = np.sum(
                        ts_students_bool[day * T : (day + 1) * T, :], axis=0
                    )
                    num_events_day_attending = num_events_day[attending]
                    delta_single = np.sum(num_events_day_attending == 0) - np.sum(
                        num_events_day_attending == 1
                    )
                    delta_last = np.sum(attending) if hour == T - 1 else 0
                    delta_consec = compute_delta_consec_insert(day, hour, attending)

                    delta_soft = delta_single + delta_last + delta_consec
                    delta_distance = -attendees[event]

                    neighbours.append([event, ts, room, delta_distance, delta_soft])

    # Extract:
    if moves[1]:
        for event in scheduled_events:
            attending = attends[:, event]
            day = ts // T
            hour = ts % T
            num_events_day = np.sum(
                ts_students_bool[day * T : (day + 1) * T, :], axis=0
            )
            num_events_day_attending = num_events_day[attending]
            delta_single = np.sum(num_events_day_attending == 2) - np.sum(
                num_events_day_attending == 1
            )
            delta_last = -np.sum(attending) if hour == T - 1 else 0
            delta_consec = compute_delta_consec_extract(day, hour, attending)

            delta_soft = delta_single + delta_last + delta_consec
            delta_distance = attendees[event]

            neighbours.append([event, -1, -1, delta_distance, delta_soft])

    # Swap:
    if moves[2]:
        for i, e1 in enumerate(scheduled_events):
            for j in range(i + 1, len(scheduled_events)):
                e2 = scheduled_events[j]
                # Condition 3
                ts1, room1 = solution[e1]
                ts2, room2 = solution[e2]

                # Condition 4
                if not (event_availability[e1, ts2] and event_availability[e2, ts1]):
                    continue

                # Condition 2
                if not (
                    room_satisfies_event[room2, e1] and room_satisfies_event[room1, e2]
                ):
                    continue

                # Condition 1
                ts1_students = ts_students_bool[ts1] & ~attends[:, e1]
                ts2_students = ts_students_bool[ts2] & ~attends[:, e2]
                if np.any(attends[:, e1] & ts2_students) or np.any(
                    attends[:, e2] & ts1_students
                ):
                    continue

                # Condition 5
                successors_e1 = np.where(before[e1])[0]
                if np.any(
                    (solution[successors_e1, 0] != -1)
                    & (solution[successors_e1, 0] < ts2)
                ):
                    continue
                successors_e2 = np.where(before[e2])[0]
                if np.any(
                    (solution[successors_e2, 0] != -1)
                    & (solution[successors_e2, 0] < ts1)
                ):
                    continue

                attending_e1_only = attends[:, e1] & ~attends[:, e2]
                attending_e2_only = attends[:, e2] & ~attends[:, e1]
                d1, h1 = ts1 // T, ts1 % T
                d2, h2 = ts2 // T, ts2 % T
                num_events_d1 = np.sum(
                    ts_students_bool[d1 * T : (d1 + 1) * T, :], axis=0
                )
                num_events_d2 = np.sum(
                    ts_students_bool[d2 * T : (d2 + 1) * T, :], axis=0
                )
                delta_single_e1 = (
                    0
                    if d1 == d2
                    else (
                        np.sum(num_events_d1[attending_e1_only] == 2)
                        - np.sum(num_events_d1[attending_e1_only] == 1)
                        + np.sum(num_events_d2[attending_e1_only] == 0)
                        - np.sum(num_events_d2[attending_e1_only] == 1)
                    )
                )
                delta_single_e2 = (
                    0
                    if d1 == d2
                    else (
                        np.sum(num_events_d2[attending_e2_only] == 2)
                        - np.sum(num_events_d2[attending_e2_only] == 1)
                        + np.sum(num_events_d1[attending_e2_only] == 0)
                        - np.sum(num_events_d1[attending_e2_only] == 1)
                    )
                )
                delta_last_e1 = np.sum(attending_e1_only) * (
                    (1 if h2 == T - 1 else 0) - (1 if h1 == T - 1 else 0)
                )
                delta_last_e2 = np.sum(attending_e2_only) * (
                    (1 if h1 == T - 1 else 0) - (1 if h2 == T - 1 else 0)
                )
                delta_consec_e1 = compute_delta_consec_extract(
                    d1, h1, attending_e1_only
                ) + compute_delta_consec_insert(d2, h2, attending_e1_only)
                delta_consec_e2 = compute_delta_consec_extract(
                    d2, h2, attending_e2_only
                ) + compute_delta_consec_insert(d1, h1, attending_e2_only)

                delta_soft = (
                    delta_single_e1
                    + delta_last_e1
                    + delta_consec_e1
                    + delta_single_e2
                    + delta_last_e2
                    + delta_consec_e2
                )
                delta_distance = 0

                neighbours.append([-1, e1, e2, delta_distance, delta_soft])

    return np.array(neighbours, dtype=np.int32)


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


def total_cost(pect: Pectp, solution: Pects) -> int:
    """
    Returns total cost of evaluation

    Args:
        pect: PECT problem
        solution: PECT solution

    Returns:
        Total cost
    """

    evaluation = evaluate(pect, solution)

    return evaluation[0] + evaluation[1]


def write_solution(path: Path, solution: Pects) -> None:
    """
    Writes a solution to file

    Args:
        path: string to file path
        solution: PECT solution
    """

    with open(path, "w", encoding="utf-8") as file:
        for timeslot, room in solution:
            file.write(f"{timeslot} {room}\n")


def write_formatted_solution(path: Path, pect: Pectp, solution: Pects) -> None:
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

    for i, event in enumerate(solution):
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


def write_evaluation(path: Path, evaluation: tuple[int, int]) -> None:
    """
    Writes a formatted evaluation to file

    Args:
        path: string to file path
        evaluation: tuple representing distance to feasibility and soft cost
    """

    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Instance: {path.stem}\n")
        file.write(f"Distance to feasibility: {evaluation[0]}\n")
        file.write(f"Soft cost: {evaluation[1]}\n")
        file.write(f"Total: {evaluation[0] + evaluation[1]}\n")
        file.write("\n")
