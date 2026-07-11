"""Randomized correctness tests: DP (tabulation & memoization) vs brute force O(2^n)."""
import random
from dp_job_scheduling import (
    weighted_job_scheduling_tabulation,
    weighted_job_scheduling_memoization,
    brute_force_optimal,
)


def random_jobs(n, seed, max_time=20, max_profit=20):
    rng = random.Random(seed)
    jobs = []
    for _ in range(n):
        s = rng.randint(0, max_time - 1)
        e = rng.randint(s + 1, max_time)
        p = rng.randint(1, max_profit)
        jobs.append((s, e, p))
    return jobs


def test_random_small_instances(trials=200, max_n=12):
    for seed in range(trials):
        n = random.Random(seed).randint(1, max_n)
        jobs = random_jobs(n, seed)
        p_tab, _ = weighted_job_scheduling_tabulation(jobs)
        p_memo, _ = weighted_job_scheduling_memoization(jobs)
        p_bf, _ = brute_force_optimal(jobs)
        assert p_tab == p_bf, f"Tabulation mismatch seed={seed}: {p_tab} vs {p_bf}, jobs={jobs}"
        assert p_memo == p_bf, f"Memoization mismatch seed={seed}: {p_memo} vs {p_bf}, jobs={jobs}"
    print(f"[PASS] DP (tabulation & memoization) match brute force on {trials} random instances (n<=12)")


def test_selected_jobs_are_non_overlapping():
    for seed in range(50):
        n = random.Random(seed + 1000).randint(1, 15)
        jobs = random_jobs(n, seed + 1000)
        _, selected = weighted_job_scheduling_tabulation(jobs)
        selected_sorted = sorted(selected, key=lambda j: j[0])
        for i in range(len(selected_sorted) - 1):
            assert selected_sorted[i][1] <= selected_sorted[i + 1][0], \
                f"Overlapping jobs selected: {selected_sorted[i]} and {selected_sorted[i+1]}"
    print("[PASS] Selected job subsets are always mutually non-overlapping")


if __name__ == "__main__":
    test_random_small_instances()
    test_selected_jobs_are_non_overlapping()
    print("\nAll DP correctness tests passed.")