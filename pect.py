"""PECT definition and methods module."""

import numpy as np
from utils import consecutive_groups, find_duplicate_indices

# PECT problem
type Pectp = tuple[
    int, int, int, int, list[int], list[int], list[int], list[int], list[int], list[int]
]
# PECT solution
type Pects = list[list[int]]


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
    sampling_rate: float = 1.0,
) -> np.ndarray:
    """
    Efficiently generates and returns neighbourhood of a solution.
    In case the neighbour is generated by swap step, array is in the form
    [event, timeslot, room, relative distance to feasibility, aprox. relative soft cost].
    In case the neighbour is generated by swap step, array is in the form
    [-1, event1, event2, relative distance to feasibility, aprox. relative soft cost]

    Args:
        np_pect: numpy PECT problem
        np_solution: numpy PECT solution
        moves: tuple of booleans (insert, extract, swap) indicating which types of moves to include in the neighbourhood
        sampling_rate: float indicating the fraction of moves to be processed

    Returns:
        Array of neighbours
    """
    (
        _,
        r,
        _,
        s,
        room_sizes,
        attends,
        roomfeatures,
        eventfeatures,
        event_availability,
        before,
    ) = np_pect

    # Consts
    t = 45
    T = 9

    # Renaming
    solution = np_solution  # (n, 2)

    # how many students per event
    attendees = np.sum(attends, axis=0, dtype=np.int32)  # (n,)

    # is room valid for event
    room_satisfies_event = (room_sizes[:, None] >= attendees[None, :]) & np.all(
        (~eventfeatures[None, :, :] | roomfeatures[:, None, :]), axis=2
    )  # (r, n)

    # scheduled/unschedules indxs and timeslots
    unscheduled_events = np.where(solution[:, 0] == -1)[0]  # (u,)
    scheduled_events = np.where(solution[:, 0] != -1)[0]  # (m,)
    timeslots = solution[:, 0]  # (n,)
    scheduled = solution[:, 0] != -1  # (n,)

    # pred and succ timeslots
    max_pred_ts = np.max(
        np.where(before & scheduled[:, None], timeslots[:, None], -1), axis=0
    )  # (n,)
    min_succ_ts = np.min(
        np.where(before & scheduled[None, :], timeslots[None, :], t), axis=1
    )  # (n,)

    # occcupied pairs
    occupied_pairs = np.zeros((t, r), dtype=np.bool_)  # (t, r)
    occupied_pairs[solution[scheduled_events, 0], solution[scheduled_events, 1]] = (
        True  # (t, r)
    )

    # student in timeslot
    ts_students_bool = np.zeros((t, s), dtype=np.bool_)  # (t, s)
    # ts_students_bool[solution[scheduled_events, 0]] |= attends[:, scheduled_events].T  # (t, s)
    # TODO: vectorize to keep it consistent with rest
    scheduled_timeslots = solution[scheduled_events, 0]
    for ts in range(t):
        events_in_ts = scheduled_events[scheduled_timeslots == ts]
        if events_in_ts.size > 0:
            ts_students_bool[ts] = np.any(attends[:, events_in_ts], axis=1)

    # events in day for student
    # pylint: disable=too-many-function-args
    day_busy = np.sum(
        ts_students_bool.reshape(5, T, -1), axis=1, dtype=np.uint8
    )  # (num_days, s)

    # Allocation
    max_neighbours = (
        len(unscheduled_events) * t * r  # Insert
        + len(scheduled_events)  # Extract
        + (len(scheduled_events) * (len(scheduled_events) - 1)) // 2  # Swap
    )
    neighbours = np.empty((max_neighbours, 5), dtype=np.int32)  # (max_neighbours, 5)
    neighbour_idx = 0

    # Approximates cost change
    # But it makes it way faster and simpler than computing exact sequence length
    def compute_delta_consec(ts, attending):
        t, s = ts_students_bool.shape
        k = len(ts)
        prev_ts = ts - 1  # (k,)
        next_ts = ts + 1  # (k,)
        has_prev = ts > 0  # (k,)
        has_next = ts < t - 1  # (k,)
        prev_status = np.zeros((k, s), dtype=bool)  # (k, s)
        next_status = np.zeros((k, s), dtype=bool)  # (k, s)
        prev_status[has_prev] = ts_students_bool[prev_ts[has_prev]]  # (k_valid, s)
        next_status[has_next] = ts_students_bool[next_ts[has_next]]  # (k_valid, s)
        contrib = np.einsum("ij,ji->ij", prev_status + next_status, attending)  # (k, s)
        result = np.sum(contrib, axis=1)  # (k,)
        return result

    # Insert, Extract, Swap

    # Insert:
    if moves[0]:
        # Sampling
        k = len(unscheduled_events)
        if k == 0:
            sampled = []
        else:
            sampled = np.random.choice(k, size=int(max(1, sampling_rate * k)))
        # Condition 4
        valid_rooms_per_event = room_satisfies_event[
            :, unscheduled_events[sampled]
        ]  # (r, u)
        # Condition 2
        valid_ts_per_event = event_availability[unscheduled_events[sampled]]  # (u, t)
        room_idx, event_idx, ts_idx = np.where(
            valid_rooms_per_event[:, :, None] & valid_ts_per_event[None, :, :]
        )
        events, timeslots, rooms = (
            unscheduled_events[sampled[event_idx]],
            ts_idx,
            room_idx,
        )  # (k,)
        # Condition 3
        occupied_mask = ~occupied_pairs[timeslots, rooms]  # (k,)
        events, timeslots, rooms = (
            events[occupied_mask],
            timeslots[occupied_mask],
            rooms[occupied_mask],
        )  # (k',)
        # Condition 1
        conflict_mask = ~np.any(
            attends[:, events].T & ts_students_bool[timeslots], axis=1
        )  # (k',)
        events, timeslots, rooms = (
            events[conflict_mask],
            timeslots[conflict_mask],
            rooms[conflict_mask],
        )  # (k'',)
        # Condition 5
        successor_mask = timeslots < min_succ_ts[events]  # (k'',)
        events, timeslots, rooms = (
            events[successor_mask],
            timeslots[successor_mask],
            rooms[successor_mask],
        )  # (k''',)
        predecessor_mask = timeslots > max_pred_ts[events]  # (k''',)
        events, timeslots, rooms = (
            events[predecessor_mask],
            timeslots[predecessor_mask],
            rooms[predecessor_mask],
        )  # (k'''',)

        # k <=> k''''
        days = timeslots // T  # (k,)
        hours = timeslots % T  # (k,)
        attending = attends[:, events]  # (s, k)

        # Slow
        contributions = np.where(
            day_busy[days, :] == 0, 1, np.where(day_busy[days, :] == 1, -1, 0)
        )  # (k, s)
        delta_single = np.einsum("ij,ji->i", contributions, attending)  # (k,)

        delta_last = attendees[events] * (hours == T - 1)  # (k,)

        delta_consec = compute_delta_consec(timeslots, attending)  # (k,)

        delta_soft = delta_single + delta_last + delta_consec  # (k,)
        delta_distance = -attendees[events]  # (k,)

        new_neighbours = np.column_stack(
            (events, timeslots, rooms, delta_distance, delta_soft)
        )  # (k, 5)
        neighbours[neighbour_idx : neighbour_idx + new_neighbours.shape[0]] = (
            new_neighbours
        )
        neighbour_idx += new_neighbours.shape[0]

    # Extract:
    if moves[1]:
        # Sampling
        k = len(scheduled_events)
        if k == 0:
            sampled = []
        else:
            sampled = np.random.choice(k, size=int(max(1, sampling_rate * k)))
        scheduled_sampled = scheduled_events[sampled]
        # All conditions
        events, timeslots, rooms = (
            scheduled_sampled,
            solution[scheduled_sampled, 0],
            solution[scheduled_sampled, 1],
        )  # (m,)
        k = len(events)

        # m <=> k
        days = timeslots // T  # (k,)
        hours = timeslots % T  # (k,)
        attending = attends[:, events]  # (s, k)

        # Slow
        contributions = np.where(
            day_busy[days, :] == 2, 1, np.where(day_busy[days, :] == 1, -1, 0)
        )  # (k, s)
        delta_single = np.einsum("ij,ji->i", contributions, attending)  # (k,)

        delta_last = -attendees[events] * (hours == T - 1)  # (k,)

        delta_consec = -compute_delta_consec(timeslots, attending)  # (k,)

        delta_soft = delta_single + delta_last + delta_consec
        delta_distance = attendees[events]  # (k,)

        new_neighbours = np.column_stack(
            (events, np.full(k, -1), np.full(k, -1), delta_distance, delta_soft)
        )  # (k, 5)

        neighbours[neighbour_idx : neighbour_idx + new_neighbours.shape[0]] = (
            new_neighbours
        )
        neighbour_idx += new_neighbours.shape[0]

    if moves[2]:
        # Condition 3
        i, j = np.triu_indices(len(scheduled_events), k=1)
        # Sampling
        k = len(i)
        if k == 0:
            sampled = []
        else:
            sampled = np.random.choice(k, size=int(max(1, sampling_rate * k)))
        e1 = scheduled_events[i[sampled]]  # Shape: (k,)
        e2 = scheduled_events[j[sampled]]  # Shape: (k,)
        ts1, r1 = solution[e1, 0], solution[e1, 1]  # (k,)
        ts2, r2 = solution[e2, 0], solution[e2, 1]  # (k,)

        # Condition 2
        avail_e1_in_ts2 = event_availability[e1, ts2]  # (k,)
        avail_e2_in_ts1 = event_availability[e2, ts1]  # (k,)
        valid_swap = avail_e1_in_ts2 & avail_e2_in_ts1
        e1, e2, ts1, ts2, r1, r2 = (
            e1[valid_swap],
            e2[valid_swap],
            ts1[valid_swap],
            ts2[valid_swap],
            r1[valid_swap],
            r2[valid_swap],
        )  # (k',)

        # Condition 4
        room_ok_e1_in_r2 = room_satisfies_event[r2, e1]  # (k',)
        room_ok_e2_in_r1 = room_satisfies_event[r1, e2]  # (k',)
        valid_swap = room_ok_e1_in_r2 & room_ok_e2_in_r1
        e1, e2, ts1, ts2, r1, r2 = (
            e1[valid_swap],
            e2[valid_swap],
            ts1[valid_swap],
            ts2[valid_swap],
            r1[valid_swap],
            r2[valid_swap],
        )  # (k'',)

        # Condition 5
        e1_pred_ok = ts2 > max_pred_ts[e1]  # (k'',)
        e1_succ_ok = ts2 < min_succ_ts[e1]  # (k'',)
        e2_pred_ok = ts1 > max_pred_ts[e2]  # (k'',)
        e2_succ_ok = ts1 < min_succ_ts[e2]  # (k'',)
        e1_e2_ok = ~(before[e1, e2] | before[e2, e1])  # (k'',)
        valid_swap = e1_pred_ok & e1_succ_ok & e2_pred_ok & e2_succ_ok & e1_e2_ok
        e1, e2, ts1, ts2, r1, r2 = (
            e1[valid_swap],
            e2[valid_swap],
            ts1[valid_swap],
            ts2[valid_swap],
            r1[valid_swap],
            r2[valid_swap],
        )  # (k''',)

        # Condition 1
        # Slow
        conflict_ts2 = np.any(
            attends[:, e1] & ~attends[:, e2] & ts_students_bool[ts2].T, axis=0
        )  # (k''',)
        conflict_ts1 = np.any(
            attends[:, e2] & ~attends[:, e1] & ts_students_bool[ts1].T, axis=0
        )  # (k''',)
        valid_swap = ~(conflict_ts2 | conflict_ts1)
        e1, e2, ts1, ts2, r1, r2 = (
            e1[valid_swap],
            e2[valid_swap],
            ts1[valid_swap],
            ts2[valid_swap],
            r1[valid_swap],
            r2[valid_swap],
        )  # (k_valid,)

        k_valid = len(e1)

        # k_valid <=> k
        days_ts1 = ts1 // T  # (k,)
        days_ts2 = ts2 // T  # (k,)
        hours_ts1 = ts1 % T  # (k,)
        hours_ts2 = ts2 % T  # (k,)
        attending_e1 = attends[:, e1]  # (s, k)
        attending_e2 = attends[:, e2]  # (s, k)
        attending_e1_only = (attending_e1 & ~attending_e2).T  # (k, s)
        attending_e2_only = (attending_e2 & ~attending_e1).T  # (k, s)

        # Slow
        delta_single = (
            -np.sum((day_busy[days_ts1, :] == 1) & attending_e1_only, axis=1)
            + np.sum((day_busy[days_ts1, :] == 2) & attending_e1_only, axis=1)
            + np.sum((day_busy[days_ts2, :] == 0) & attending_e1_only, axis=1)
            - np.sum((day_busy[days_ts2, :] == 1) & attending_e1_only, axis=1)
            + np.sum((day_busy[days_ts1, :] == 0) & attending_e2_only, axis=1)
            - np.sum((day_busy[days_ts1, :] == 1) & attending_e2_only, axis=1)
            - np.sum((day_busy[days_ts2, :] == 1) & attending_e2_only, axis=1)
            + np.sum((day_busy[days_ts2, :] == 2) & attending_e2_only, axis=1)
        )  # (k,)

        delta_last_e1 = attendees[e1] * (
            (hours_ts2 == T - 1).astype(np.int8) - (hours_ts1 == T - 1).astype(np.int8)
        )  # (k,)
        delta_last_e2 = attendees[e2] * (
            (hours_ts1 == T - 1).astype(np.int8) - (hours_ts2 == T - 1).astype(np.int8)
        )  # (k,)

        # Pretty slow
        # Approximation
        delta_consec_e1 = compute_delta_consec(
            ts2, attending_e1
        ) - compute_delta_consec(
            ts1, attending_e1
        )  # (k,)
        delta_consec_e2 = compute_delta_consec(
            ts1, attending_e2
        ) - compute_delta_consec(
            ts2, attending_e2
        )  # (k,)

        new_neighbours = np.column_stack(
            (
                np.full(k_valid, -1, dtype=np.int32),
                e1,
                e2,
                np.zeros(k_valid, dtype=np.int32),
                delta_single
                + delta_last_e1
                + delta_last_e2
                + delta_consec_e1
                + delta_consec_e2,
            )
        )  # (k, 5)

        neighbours[neighbour_idx : neighbour_idx + k_valid] = new_neighbours
        neighbour_idx += k_valid

    return neighbours[:neighbour_idx]


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
