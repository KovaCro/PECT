""" Utility functions module. """


def consecutive_groups(numbers: list[int]) -> list[list[int]]:
    """
    Splits list of numbers into consecutive groups

    Args:
      numbers: list of numbers

    Returns:
      list of consecutive groups
    """

    if not numbers:
        return []

    group = [numbers[0]]
    groups = []
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] == 1:
            group.append(numbers[i])
        else:
            groups.append(group)
            group = [numbers[i]]
    groups.append(group)

    return groups


def find_duplicate_indices(lst: list[int]) -> list[int]:
    """
    Finds the indices of duplicate elements in a list

    Args:
      lst: input list

    Returns:
      list of duplicate indices
    """

    seen = set()
    duplicates = []
    for i, num in enumerate(lst):
        if num in seen:
            duplicates.append(i)
        else:
            seen.add(num)

    return duplicates
