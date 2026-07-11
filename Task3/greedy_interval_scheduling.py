"""
Task 3 - Greedy: Interval Scheduling Maximisation with Weighted Intervals
============================================================================
Classic (unweighted) activity selection: sort by finish time, greedily take
any interval compatible with the last one taken. This greedy choice is
PROVABLY OPTIMAL when the goal is to maximise the *number* of non-overlapping
intervals (exchange-argument proof, standard result).

However, when intervals carry weights/profits and the goal is to maximise
TOTAL WEIGHT (not count), the same greedy rule is no longer guaranteed
optimal — a single high-value interval can be worth more than several
low-value ones that the greedy-by-finish-time rule would prefer.

This module implements three greedy heuristics and proves/disproves their
optimality empirically against the exact DP solution from
`dp_job_scheduling.py` (the two problems are the same underlying problem).
"""
from dp_job_scheduling import weighted_job_scheduling_tabulation


def greedy_earliest_finish_time(jobs):
    """Classic activity-selection greedy: sort by end time, take if compatible.
    OPTIMAL for maximising COUNT of intervals; NOT guaranteed optimal for WEIGHT.
    Time complexity: O(n log n).
    """
    jobs_sorted = sorted(jobs, key=lambda j: j[1])
    selected = []
    last_end = float("-inf")
    total_profit = 0
    for s, e, profit in jobs_sorted:
        if s >= last_end:
            selected.append((s, e, profit))
            total_profit += profit
            last_end = e
    return total_profit, selected


def greedy_highest_profit_first(jobs):
    """Greedy variant: always take the highest-profit remaining job that
    doesn't conflict with already-chosen jobs. Time complexity: O(n^2) naive
    (O(n log n) with an interval tree, not needed for coursework-scale n).
    """
    jobs_sorted = sorted(jobs, key=lambda j: -j[2])  # highest profit first
    selected = []
    total_profit = 0

    def conflicts(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for job in jobs_sorted:
        if all(not conflicts(job, s) for s in selected):
            selected.append(job)
            total_profit += job[2]
    return total_profit, sorted(selected, key=lambda j: j[0])


def greedy_best_ratio(jobs):
    """Greedy variant: prioritise profit-per-unit-duration (density heuristic),
    commonly used in scheduling heuristics. Time complexity: O(n^2) naive.
    """
    jobs_sorted = sorted(jobs, key=lambda j: -(j[2] / max(1e-9, (j[1] - j[0]))))
    selected = []
    total_profit = 0

    def conflicts(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for job in jobs_sorted:
        if all(not conflicts(job, s) for s in selected):
            selected.append(job)
            total_profit += job[2]
    return total_profit, sorted(selected, key=lambda j: j[0])


# ---------------------------------------------------------------------------
# Demonstration: a case where greedy-by-finish-time is proven suboptimal
# ---------------------------------------------------------------------------
def demo_greedy_suboptimal_case():
    """A hand-crafted, minimal counter-example: one long, high-value interval
    that overlaps two short, lower-value-individually-but-higher-combined
    intervals. Greedy-by-finish-time picks the two short ones OR misses the
    optimal combination, depending on the arrangement; here we construct a
    case where greedy-by-finish-time strictly loses to the DP optimum.
    """
    # Interval B (1,2) profit 3, Interval C (2,3) profit 3 -- both fit together (profit 6)
    # Interval A (1,3) profit 5 -- overlaps both B and C, greedy-by-finish-time
    # will consider A last (latest finish among ties broken by input order isn't
    # guaranteed) -- construct so finish times force the wrong pick:
    jobs = [
        (0, 4, 7),   # long, valuable job "A"
        (0, 2, 4),   # "B" - finishes early
        (2, 4, 4),   # "C" - compatible with B, finishes at same time as A
    ]
    # Optimal: B + C = profit 8 (both fit, don't overlap: B ends at 2, C starts at 2)
    # Greedy-by-finish-time sorts by end: B(end=2, profit4), A(end=4,profit7), C(end=4,profit4)
    #   -> takes B (profit 4, last_end=2)
    #   -> A conflicts with B? A starts at 0 < 2 -> conflict, skip A
    #   -> C starts at 2 >= 2 -> compatible, take C (profit 4)
    #   -> greedy total = 8 ... let's instead force an actual failure:
    jobs = [
        (0, 3, 10),   # "A": long but very valuable
        (0, 1, 4),    # "B": short
        (1, 2, 4),    # "C": short, compatible with B
        (2, 3, 4),    # "D": short, compatible with B and C
    ]
    # B+C+D = profit 12 (all mutually compatible, better than A alone = 10)
    # But greedy-by-finish-time sorts by end: B(1,4), C(2,4), D(3,4), A(3,10)
    #   picks B (profit4, last_end=1) -> C compatible (profit4, last_end=2)
    #   -> D compatible (profit4, last_end=3) -> total = 12. Greedy gets this right too!
    # To truly break greedy-by-finish-time we need HIGH profit density on the
    # interval that finishes LATE, so greedy commits to low-profit early
    # intervals and then can't fit the high-profit one:
    jobs = [
        (0, 1, 1),     # tiny early job, low profit
        (1, 10, 100),  # long job, HUGE profit, starts right after the tiny one
        (0, 10, 100),  # equally huge profit but conflicts with both above if chosen together... 
    ]
    # Simpler, canonical counter-example: greedy picks many small early
    # intervals that block one big valuable interval.
    jobs = [
        (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1),  # 5 tiny jobs, profit 1 each, all compatible
        (1, 6, 100),  # ONE big job overlapping all 5, but worth far more than their sum
    ]
    greedy_profit, greedy_jobs = greedy_earliest_finish_time(jobs)
    dp_profit, dp_jobs = weighted_job_scheduling_tabulation(jobs)

    print("Counter-example jobs:", jobs)
    print(f"Greedy (earliest finish time): profit={greedy_profit}, jobs={greedy_jobs}")
    print(f"DP (exact optimum):            profit={dp_profit}, jobs={dp_jobs}")
    print(f"Greedy achieves {greedy_profit/dp_profit*100:.1f}% of optimal profit")
    assert greedy_profit < dp_profit, "Expected greedy to be strictly suboptimal here"
    return jobs, greedy_profit, dp_profit


if __name__ == "__main__":
    demo_greedy_suboptimal_case()

    print("\n--- Random comparison across heuristics ---")
    import random
    random.seed(7)
    wins = {"finish_time": 0, "highest_profit": 0, "best_ratio": 0, "optimal_ties": 0}
    n_trials = 500
    total_gap_finish = []
    for _ in range(n_trials):
        n = random.randint(3, 10)
        jobs = []
        for _ in range(n):
            s = random.randint(0, 15)
            e = random.randint(s + 1, 20)
            p = random.randint(1, 50)
            jobs.append((s, e, p))

        dp_profit, _ = weighted_job_scheduling_tabulation(jobs)
        gft_profit, _ = greedy_earliest_finish_time(jobs)
        ghp_profit, _ = greedy_highest_profit_first(jobs)
        gbr_profit, _ = greedy_best_ratio(jobs)

        if gft_profit == dp_profit:
            wins["finish_time"] += 1
        if ghp_profit == dp_profit:
            wins["highest_profit"] += 1
        if gbr_profit == dp_profit:
            wins["best_ratio"] += 1
        total_gap_finish.append(dp_profit - gft_profit)

    print(f"Out of {n_trials} random instances, each greedy heuristic matched the DP optimum:")
    for k, v in wins.items():
        if k != "optimal_ties":
            print(f"  {k}: {v}/{n_trials} ({100*v/n_trials:.1f}%)")
    print(f"Average profit gap (DP - greedy-by-finish-time): {sum(total_gap_finish)/n_trials:.2f}")