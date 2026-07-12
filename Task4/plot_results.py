import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results.csv")))
ns = [int(r["n"]) for r in rows]
lb = [int(r["lower_bound"]) for r in rows]
ffd_bins = [float(r["ffd_bins_mean"]) for r in rows]
sa_bins = [float(r["sa_bins_mean"]) for r in rows]
ffd_gap = [float(r["ffd_gap_pct"]) for r in rows]
sa_gap = [float(r["sa_gap_pct"]) for r in rows]
ffd_time = [float(r["ffd_time_mean"]) * 1000 for r in rows]
sa_time = [float(r["sa_time_mean"]) * 1000 for r in rows]

# ---------------------------------------------------------------------------
# Fig 1: Bins used vs lower bound
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(ns, lb, "k--", label="Theoretical lower bound")
ax1.plot(ns, ffd_bins, "o-", label="FFD (greedy)")
ax1.plot(ns, sa_bins, "s-", label="SA (metaheuristic)")
ax1.set_xlabel("n (number of items)")
ax1.set_ylabel("Bins used (mean of 5 trials)")
ax1.set_title("Solution Quality vs Lower Bound")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(ns, ffd_gap, "o-", label="FFD % over lower bound")
ax2.plot(ns, sa_gap, "s-", label="SA % over lower bound")
ax2.set_xlabel("n (number of items)")
ax2.set_ylabel("Optimality gap (%)")
ax2.set_title("Optimality Gap vs Problem Size")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("Multi-Dimensional Bin Packing: FFD vs Simulated Annealing\n"
             "SA consistently matches or beats FFD, at the cost of far more computation")
plt.tight_layout()
plt.savefig("fig1_quality_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Fig 2: Runtime comparison
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(ns, ffd_time, "o-", label="FFD (greedy construction)")
plt.plot(ns, sa_time, "s-", label="SA (2000 iterations)")
plt.yscale("log")
plt.xlabel("n (number of items)")
plt.ylabel("Wall-clock time (ms, mean of 5 trials)")
plt.title("Runtime: FFD vs SA\nSA's iterative local search costs orders of magnitude more time for its quality gain")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_runtime_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Fig 3: SA convergence trace (cost vs iteration) for one representative run
# ---------------------------------------------------------------------------
trace_rows = list(csv.DictReader(open("sa_convergence_trace.csv")))
iterations = [int(r["iteration"]) for r in trace_rows]
best_cost = [int(r["best_cost_so_far"]) for r in trace_rows]

plt.figure(figsize=(8, 5))
plt.plot(iterations, best_cost, "-", color="tab:purple", linewidth=1.5)
plt.xlabel("SA iteration")
plt.ylabel("Best bins-used found so far")
plt.title("Simulated Annealing Convergence (n=80 items)\n"
          "Best-so-far cost only decreases (by construction) but individual moves may increase local cost")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_sa_convergence.png", dpi=150)
plt.close()

print("Saved fig1_quality_comparison.png, fig2_runtime_comparison.png, fig3_sa_convergence.png")