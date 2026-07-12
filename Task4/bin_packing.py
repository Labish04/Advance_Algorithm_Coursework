"""
Task 4: NP-Hard Problem — Multi-Dimensional Bin Packing (MDBP)
=================================================================
Problem: Given n items, each with a resource demand vector in d dimensions
(e.g. weight in kg, volume in m^3), and bins each with capacity vector C
(same in every dimension for every bin, homogeneous bins), pack ALL items
into the MINIMUM number of bins such that for every bin and every dimension,
the sum of packed items' demand in that dimension does not exceed capacity.

NP-Hardness (sketch)
---------------------
Even the 1-dimensional special case (classic Bin Packing) is NP-hard, shown
by a reduction from PARTITION (NP-complete): given a multiset of positive
integers S with total sum T, PARTITION asks whether S can be split into two
subsets each summing to T/2. This is equivalent to: "can the items in S be
packed into 2 bins of capacity T/2?" A polynomial-time algorithm for 1D bin
packing (decision version: "can these items fit into k bins?") would solve
PARTITION in polynomial time by asking k=2. Since PARTITION is NP-complete,
1D Bin Packing is NP-hard. Multi-dimensional Bin Packing strictly generalises
the 1D case (set d=1) and is therefore *at least* as hard -- also NP-hard.
No known algorithm solves MDBP exactly in polynomial time, and none is
believed to exist unless P = NP (see Advance_Algorithm_Lecture.md Sec 1.4-1.5).

This module provides:
    - Item / Bin representation
    - A feasibility checker
    - A theoretical LOWER BOUND on the number of bins needed (for measuring
      the optimality gap of heuristic solutions without needing an exact
      exponential solver at realistic problem sizes)
    - A brute-force EXACT solver (only usable for tiny n, used to validate
      heuristics on small instances)
"""
import math
import itertools
import random


class BinPackingInstance:
    def __init__(self, items, capacity):
        """items: list of tuples, each a d-dimensional demand vector.
        capacity: tuple, the (homogeneous) per-bin capacity vector, same length as items[i]."""
        self.items = items
        self.capacity = capacity
        self.n = len(items)
        self.d = len(capacity)

    def lower_bound(self):
        """Continuous relaxation lower bound: for each dimension k, sum of
        demand_k / capacity_k gives a lower bound on bins needed just to fit
        that dimension's total volume; the true lower bound is the max over
        dimensions, rounded up. This is provably a valid lower bound (no
        feasible packing can use fewer bins), though it is not always tight.
        """
        best = 1
        for k in range(self.d):
            total_k = sum(item[k] for item in self.items)
            bins_k = math.ceil(total_k / self.capacity[k])
            best = max(best, bins_k)
        return best


def solution_is_feasible(instance: BinPackingInstance, assignment):
    """assignment: list, assignment[i] = bin index for item i.
    Returns True if every bin's total demand is within capacity in every dimension."""
    bins_used = set(assignment)
    for b in bins_used:
        totals = [0] * instance.d
        for i, bin_idx in enumerate(assignment):
            if bin_idx == b:
                for k in range(instance.d):
                    totals[k] += instance.items[i][k]
        for k in range(instance.d):
            if totals[k] > instance.capacity[k] + 1e-9:
                return False
    return True


def num_bins_used(assignment):
    return len(set(assignment))


def brute_force_optimal(instance: BinPackingInstance, max_bins=None):
    """EXACT solver via exhaustive search over all assignments of items to
    bins (bins limited to at most max_bins, default = n, i.e. worst case one
    item per bin). Only tractable for very small n (n <= ~8).
    Time complexity: O(k^n) where k = number of candidate bin labels -- this
    is precisely why MDBP is NP-hard: no known algorithm avoids this
    exponential blowup in the worst case.
    """
    n = instance.n
    max_bins = max_bins or n
    best = None
    # Try increasing bin counts k = lower_bound, lower_bound+1, ... until feasible found
    lb = instance.lower_bound()
    for k in range(lb, max_bins + 1):
        for assignment in itertools.product(range(k), repeat=n):
            if len(set(assignment)) > k:
                continue
            if solution_is_feasible(instance, assignment):
                return list(assignment), k
    return None, None


def generate_instance(n_items, d=2, seed=0, capacity=100, item_range=(10, 40)):
    rng = random.Random(seed)
    items = [tuple(rng.randint(*item_range) for _ in range(d)) for _ in range(n_items)]
    cap = tuple(capacity for _ in range(d))
    return BinPackingInstance(items, cap)


if __name__ == "__main__":
    inst = generate_instance(6, d=2, seed=1, capacity=50, item_range=(10, 30))
    print("Items:", inst.items)
    print("Capacity:", inst.capacity)
    print("Lower bound on bins:", inst.lower_bound())
    assignment, k = brute_force_optimal(inst)
    print(f"Brute-force optimal: {k} bins, assignment={assignment}")