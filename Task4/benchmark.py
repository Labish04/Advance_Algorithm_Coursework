"""
Benchmarks FFD (greedy) vs SA (metaheuristic) across increasing problem
sizes, measuring:
    - bins used (solution quality) vs the theoretical lower bound
    - wall-clock runtime
Also records SA's convergence trace (cost vs iteration) for the required
optimization-process visualisation.
"""
import time
import csv
import statistics as stats
from bin_packing import generate_instance, num_bins_used, solution_is_feasible
from heuristics import first_fit_decreasing, simulated_annealing

N_VALUES = [10, 20, 40, 80, 150]
N_TRIALS = 5


def run_all():
    rows = []
    convergence_trace = None  # keep one representative trace for plotting

    for n in N_VALUES:
        ffd_bins_list, sa_bins_list, ffd_times, sa_times = [], [], [], []
        for seed in range(N_TRIALS):
            inst = generate_instance(n_items=n, d=3, seed=seed, capacity=100, item_range=(10, 45))
            lb = inst.lower_bound()

            t0 = time.perf_counter()
            ffd = first_fit_decreasing(inst)
            t_ffd = time.perf_counter() - t0
            ffd_bins = num_bins_used(ffd)
            assert solution_is_feasible(inst, ffd)

            t0 = time.perf_counter()
            sa_assignment, sa_bins, history = simulated_annealing(
                inst, ffd, iterations=2000, T0=8.0, cooling=0.997, seed=seed)
            t_sa = time.perf_counter() - t0
            assert solution_is_feasible(inst, sa_assignment)

            ffd_bins_list.append(ffd_bins)
            sa_bins_list.append(sa_bins)
            ffd_times.append(t_ffd)
            sa_times.append(t_sa)

            if n == 40 and seed == 0:
                convergence_trace = (history, lb, ffd_bins)

        row = dict(
            n=n,
            lower_bound=lb,
            ffd_bins_mean=stats.mean(ffd_bins_list),
            sa_bins_mean=stats.mean(sa_bins_list),
            ffd_time_mean=stats.mean(ffd_times),
            sa_time_mean=stats.mean(sa_times),
            ffd_gap_pct=100 * (stats.mean(ffd_bins_list) - lb) / lb,
            sa_gap_pct=100 * (stats.mean(sa_bins_list) - lb) / lb,
        )
        rows.append(row)
        print(f"n={n:4d}  LB={lb:3d}  FFD={row['ffd_bins_mean']:.1f} bins ({row['ffd_gap_pct']:.1f}% over LB, "
              f"{row['ffd_time_mean']*1000:.2f}ms)  "
              f"SA={row['sa_bins_mean']:.1f} bins ({row['sa_gap_pct']:.1f}% over LB, "
              f"{row['sa_time_mean']*1000:.2f}ms)")

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results.csv")

    if convergence_trace:
        history, lb, ffd_bins = convergence_trace
        with open("sa_convergence_trace.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "best_cost_so_far"])
            for i, c in enumerate(history):
                writer.writerow([i, c])
        print(f"Saved sa_convergence_trace.csv (n=40 example, FFD start={ffd_bins}, lower_bound={lb})")

    return rows


if __name__ == "__main__":
    run_all()