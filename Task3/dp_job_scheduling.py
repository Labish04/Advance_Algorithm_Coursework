"""
Task 3 - Dynamic Programming: Weighted Job Scheduling with Time Windows
=========================================================================
Problem: Given n jobs, each with (start, end, profit), select a subset of
non-overlapping jobs (two jobs overlap if they share any time, i.e. job i
and job j conflict unless end_i <= start_j or end_j <= start_i) that
maximises total profit.

Subproblem definition & recurrence relation
--------------------------------------------
1. Sort jobs by end time ascending: job[0], job[1], ..., job[n-1].
2. p(i) = the largest index j < i such that job[j].end <= job[i].start
   (the latest job that does NOT conflict with job i). Found via binary
   search since ends are sorted -> O(log n) per job.
3. Let dp[i] = maximum profit achievable using only jobs[0..i-1] (the
   first i jobs in end-time order).
   Recurrence:
       dp[0] = 0
       dp[i] = max( dp[i-1],                      # skip job i-1
                     profit[i-1] + dp[p(i-1)+1] )  # take job i-1
   dp[n] is the answer.
4. Base case: dp[0] = 0 (no jobs -> zero profit).

Two implementations are provided: top-down memoisation and bottom-up
tabulation, plus solution reconstruction (which jobs were actually chosen).

Time Complexity : O(n log n)  (sorting + binary search for p(i) at each step)
Space Complexity: O(n)        (dp table + recursion stack for memo version)
"""
from bisect import bisect_right
from functools import lru_cache


def _prepare(jobs):
    """jobs: list of (start, end, profit). Returns sorted jobs and p(i) array."""
    jobs_sorted = sorted(jobs, key=lambda j: j[1])  # sort by end time
    ends = [j[1] for j in jobs_sorted]
    p = []  # p[i] = index of latest job compatible with jobs_sorted[i], or -1
    for i, (s, e, prof) in enumerate(jobs_sorted):
        # find latest job whose end <= s (this job's start) via binary search
        idx = bisect_right(ends, s, 0, i) - 1
        p.append(idx)
    return jobs_sorted, p


def weighted_job_scheduling_tabulation(jobs):
    """Bottom-up DP. Returns (max_profit, selected_jobs)."""
    if not jobs:
        return 0, []
    jobs_sorted, p = _prepare(jobs)
    n = len(jobs_sorted)
    dp = [0] * (n + 1)
    choice = [False] * (n + 1)  # choice[i] True if job i-1 was taken in the optimal dp[i]

    for i in range(1, n + 1):
        s, e, profit = jobs_sorted[i - 1]
        take = profit + dp[p[i - 1] + 1]
        skip = dp[i - 1]
        if take > skip:
            dp[i] = take
            choice[i] = True
        else:
            dp[i] = skip
            choice[i] = False

    # Reconstruct selected jobs by walking choice[] backwards
    selected = []
    i = n
    while i > 0:
        if choice[i]:
            selected.append(jobs_sorted[i - 1])
            i = p[i - 1] + 1
        else:
            i -= 1
    selected.reverse()
    return dp[n], selected


def weighted_job_scheduling_memoization(jobs):
    """Top-down DP with memoisation. Returns (max_profit, selected_jobs)."""
    if not jobs:
        return 0, []
    jobs_sorted, p = _prepare(jobs)
    n = len(jobs_sorted)

    memo = {}

    def solve(i):
        # max profit considering jobs_sorted[0..i-1] (i jobs remain to be decided)
        if i == 0:
            return 0
        if i in memo:
            return memo[i]
        s, e, profit = jobs_sorted[i - 1]
        take = profit + solve(p[i - 1] + 1)
        skip = solve(i - 1)
        memo[i] = max(take, skip)
        return memo[i]

    best = solve(n)

    # Reconstruct
    selected = []
    i = n
    while i > 0:
        s, e, profit = jobs_sorted[i - 1]
        take = profit + solve(p[i - 1] + 1)
        skip = solve(i - 1)
        if take >= skip and take == memo.get(i, solve(i)):
            selected.append(jobs_sorted[i - 1])
            i = p[i - 1] + 1
        else:
            i -= 1
    selected.reverse()
    return best, selected


def brute_force_optimal(jobs):
    """O(2^n) exhaustive search — used only to validate correctness on small inputs."""
    n = len(jobs)
    best_profit = 0
    best_subset = []

    def conflicts(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    def rec(i, chosen, profit):
        nonlocal best_profit, best_subset
        if i == n:
            if profit > best_profit:
                best_profit = profit
                best_subset = list(chosen)
            return
        # skip
        rec(i + 1, chosen, profit)
        # take, if no conflict with chosen
        if all(not conflicts(jobs[i], c) for c in chosen):
            chosen.append(jobs[i])
            rec(i + 1, chosen, profit + jobs[i][2])
            chosen.pop()

    rec(0, [], 0)
    return best_profit, best_subset


if __name__ == "__main__":
    # Worked example
    # (start, end, profit)
    jobs = [
        (1, 3, 5),
        (2, 5, 6),
        (4, 6, 5),
        (6, 7, 4),
        (5, 8, 11),
        (7, 9, 2),
    ]
    profit_tab, jobs_tab = weighted_job_scheduling_tabulation(jobs)
    profit_memo, jobs_memo = weighted_job_scheduling_memoization(jobs)
    profit_bf, jobs_bf = brute_force_optimal(jobs)

    print("Tabulation:  max profit =", profit_tab, " jobs =", jobs_tab)
    print("Memoization: max profit =", profit_memo, " jobs =", jobs_memo)
    print("Brute force: max profit =", profit_bf, " jobs =", jobs_bf)
    assert profit_tab == profit_memo == profit_bf
    print("\nAll three methods agree on optimal profit:", profit_tab)