"""
Task 4 — Two Heuristics for Multi-Dimensional Bin Packing
============================================================
Heuristic 1: FIRST-FIT DECREASING (FFD)
    A classic GREEDY constructive heuristic. Items are sorted by decreasing
    "size" (here: sum across dimensions, a common aggregation for MDBP),
    then each item is placed into the FIRST bin it fits in; if it fits in no
    open bin, a new bin is opened.
    - No optimality guarantee (matches the lecture definition of "Heuristic
      Algorithm", Sec 1.7.2), but the well-known worst-case *approximation-
      style* bound for 1D FFD is 11/9 * OPT + 1 bins; no such bound is
      guaranteed to carry over exactly to the multi-dimensional case, but FFD
      remains a very fast, reasonable baseline in practice.
    - Time complexity: O(n log n) for sorting + O(n * bins) for placement
      = O(n^2) worst case (bins <= n).

Heuristic 2: SIMULATED ANNEALING (SA)
    A METAHEURISTIC local-search improvement procedure (Sec 1.9 of the
    lecture). Starts from the FFD solution and repeatedly attempts to move
    a random item to a different (possibly new) bin, accepting the move
    immediately if it doesn't increase the number of bins used, and
    accepting bin-count-increasing moves with probability e^(-delta/T) to
    escape local optima, cooling T over time.
    - No optimality guarantee either, but explores a much larger part of the
      solution space than a single greedy pass, and in practice finds
      packings using fewer bins than FFD alone (empirically verified below).
    - Time complexity: O(iterations * n) (each iteration recomputes bin
      loads for feasibility checking in O(n) worst case).
"""
import math
import random
from bin_packing import BinPackingInstance, solution_is_feasible, num_bins_used


# ---------------------------------------------------------------------------
# HEURISTIC 1: First-Fit Decreasing (greedy constructive)
# ---------------------------------------------------------------------------
def first_fit_decreasing(instance: BinPackingInstance):
    n, d = instance.n, instance.d
    order = sorted(range(n), key=lambda i: -sum(instance.items[i]))
    bin_loads = []  # list of [load_dim0, load_dim1, ...] per open bin
    assignment = [-1] * n

    for i in order:
        item = instance.items[i]
        placed = False
        for b, load in enumerate(bin_loads):
            if all(load[k] + item[k] <= instance.capacity[k] for k in range(d)):
                for k in range(d):
                    load[k] += item[k]
                assignment[i] = b
                placed = True
                break
        if not placed:
            bin_loads.append(list(item))
            assignment[i] = len(bin_loads) - 1

    return assignment


# ---------------------------------------------------------------------------
# HEURISTIC 2: Simulated Annealing (metaheuristic local search)
# ---------------------------------------------------------------------------
def compute_bin_loads(instance: BinPackingInstance, assignment):
    bins_used = sorted(set(assignment))
    loads = {b: [0] * instance.d for b in bins_used}
    for i, b in enumerate(assignment):
        for k in range(instance.d):
            loads[b][k] += instance.items[i][k]
    return loads


def cost(assignment):
    """Objective: minimise number of bins used."""
    return num_bins_used(assignment)


def simulated_annealing(instance: BinPackingInstance, initial_assignment,
                         T0=10.0, cooling=0.995, iterations=5000, seed=0):
    """Two move types are mixed:
    (a) random relocate  - move a random item to a random (possibly new) bin;
        classic SA move, provides general exploration.
    (b) targeted-empty   - pick the LEAST-LOADED bin and try to relocate ALL
        its items into other existing bins; if every item can be
        successfully re-homed, the bin count drops by 1. This targeted move
        is what actually drives bin-count reduction in practice (a purely
        random move rarely stumbles onto a full-bin-emptying sequence by
        chance) -- mixing a domain-specific move into the metaheuristic
        while keeping the same acceptance criterion is standard SA practice
        for bin-packing-style problems.
    """
    rng = random.Random(seed)
    n, d = instance.n, instance.d

    current = list(initial_assignment)
    current_cost = cost(current)
    best = list(current)
    best_cost = current_cost
    T = T0
    history = [best_cost]

    def relabel(assignment):
        used = sorted(set(assignment))
        remap = {b: idx for idx, b in enumerate(used)}
        return [remap[b] for b in assignment]

    for it in range(iterations):
        move_type = "targeted" if it % 3 == 0 else "random"
        trial = None

        if move_type == "targeted":
            loads = compute_bin_loads(instance, current)
            if len(loads) > 1:
                # pick the least-loaded bin (by total across dims) to try to empty
                least_bin = min(loads, key=lambda b: sum(loads[b]))
                items_in_bin = [i for i, b in enumerate(current) if b == least_bin]
                other_bins = [b for b in loads if b != least_bin]

                trial_loads = {b: list(loads[b]) for b in other_bins}
                trial_assignment = list(current)
                success = True
                for i in items_in_bin:
                    item = instance.items[i]
                    placed = False
                    for b in other_bins:
                        if all(trial_loads[b][k] + item[k] <= instance.capacity[k] for k in range(d)):
                            for k in range(d):
                                trial_loads[b][k] += item[k]
                            trial_assignment[i] = b
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                if success:
                    trial = relabel(trial_assignment)

        if trial is None:
            # random relocate move
            i = rng.randrange(n)
            old_bin = current[i]
            existing_bins = sorted(set(current))
            candidate_bins = existing_bins + [max(existing_bins) + 1]
            new_bin = rng.choice(candidate_bins)
            if new_bin == old_bin:
                history.append(best_cost)
                T *= cooling
                continue
            cand = list(current)
            cand[i] = new_bin
            if not solution_is_feasible(instance, cand):
                history.append(best_cost)
                T *= cooling
                continue
            trial = relabel(cand)

        trial_cost = cost(trial)
        delta = trial_cost - current_cost

        if delta <= 0:
            current, current_cost = trial, trial_cost
        else:
            if rng.random() < math.exp(-delta / max(T, 1e-9)):
                current, current_cost = trial, trial_cost

        if current_cost < best_cost:
            best, best_cost = list(current), current_cost

        T *= cooling
        history.append(best_cost)

    return best, best_cost, history


if __name__ == "__main__":
    from bin_packing import generate_instance
    inst = generate_instance(30, d=2, seed=1, capacity=100, item_range=(10, 50))

    ffd_assignment = first_fit_decreasing(inst)
    print("FFD bins used:", num_bins_used(ffd_assignment), "feasible:", solution_is_feasible(inst, ffd_assignment))

    sa_assignment, sa_cost, history = simulated_annealing(inst, ffd_assignment, iterations=3000)
    print("SA bins used:", sa_cost, "feasible:", solution_is_feasible(inst, sa_assignment))

    print("Lower bound:", inst.lower_bound())