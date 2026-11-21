from typing import List, Tuple


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged


# --- Assertions ---

# 1. Empty input
assert merge_intervals([]) == []

# 2. Non-overlapping intervals
assert merge_intervals([(1, 2), (4, 6)]) == [(1, 2), (4, 6)]

# 3. Overlapping intervals
assert merge_intervals([(1, 4), (3, 6)]) == [(1, 6)]

# 4. Adjacent intervals
assert merge_intervals([(1, 2), (2, 3)]) == [(1, 3)]

# 5. Nested intervals
assert merge_intervals([(1, 10), (3, 5)]) == [(1, 10)]

print("All 5 assertions passed!")
