"""Correctness tests for Task 4: feasibility checks + comparison against
brute-force optimal on small instances."""
from bin_packing import generate_instance, brute_force_optimal, solution_is_feasible, num_bins_used
from heuristics import first_fit_decreasing, simulated_annealing


def test_ffd_always_feasible(trials=100):
    for seed in range(trials):
        inst = generate_instance(n_items=20, d=2, seed=seed, capacity=100, item_range=(5, 60))
        assignment = first_fit_decreasing(inst)
        assert solution_is_feasible(inst, assignment), f"FFD produced infeasible solution, seed={seed}"
        assert len(assignment) == inst.n
    print(f"[PASS] FFD always produces a feasible solution ({trials} random instances)")


def test_sa_always_feasible(trials=30):
    for seed in range(trials):
        inst = generate_instance(n_items=20, d=2, seed=seed, capacity=100, item_range=(5, 60))
        ffd = first_fit_decreasing(inst)
        sa_assignment, sa_cost, _ = simulated_annealing(inst, ffd, iterations=500, seed=seed)
        assert solution_is_feasible(inst, sa_assignment), f"SA produced infeasible solution, seed={seed}"
        assert num_bins_used(sa_assignment) == sa_cost
    print(f"[PASS] SA always produces a feasible solution ({trials} random instances)")


def test_sa_never_worse_than_ffd(trials=30):
    """SA starts FROM the FFD solution and only accepts equal-or-better bests,
    so its returned best must never use MORE bins than FFD did."""
    for seed in range(trials):
        inst = generate_instance(n_items=25, d=2, seed=seed, capacity=100, item_range=(5, 60))
        ffd = first_fit_decreasing(inst)
        ffd_cost = num_bins_used(ffd)
        _, sa_cost, _ = simulated_annealing(inst, ffd, iterations=800, seed=seed)
        assert sa_cost <= ffd_cost, f"SA regressed vs FFD: {sa_cost} > {ffd_cost} (seed={seed})"
    print(f"[PASS] SA never regresses below the FFD starting point ({trials} instances)")


def test_heuristics_match_or_near_optimal_small_instances(trials=15):
    """Compare against the exact brute-force optimum on tiny instances (n<=7)."""
    exact_matches_ffd = 0
    exact_matches_sa = 0
    for seed in range(trials):
        inst = generate_instance(n_items=6, d=2, seed=seed, capacity=50, item_range=(10, 30))
        opt_assignment, opt_k = brute_force_optimal(inst, max_bins=6)
        assert opt_k is not None

        ffd = first_fit_decreasing(inst)
        ffd_k = num_bins_used(ffd)
        sa_assignment, sa_k, _ = simulated_annealing(inst, ffd, iterations=500, seed=seed)

        assert ffd_k >= opt_k, "FFD used fewer bins than the proven optimum -- impossible, bug!"
        assert sa_k >= opt_k, "SA used fewer bins than the proven optimum -- impossible, bug!"
        exact_matches_ffd += (ffd_k == opt_k)
        exact_matches_sa += (sa_k == opt_k)

    print(f"[PASS] On {trials} tiny instances (n=6, verified against exact brute force):")
    print(f"       FFD matched optimal: {exact_matches_ffd}/{trials}")
    print(f"       SA  matched optimal: {exact_matches_sa}/{trials}")


if __name__ == "__main__":
    test_ffd_always_feasible()
    test_sa_always_feasible()
    test_sa_never_worse_than_ffd()
    test_heuristics_match_or_near_optimal_small_instances()
    print("\nAll Task 4 correctness tests passed.")