import matplotlib.pyplot as plt
from dp_job_scheduling import weighted_job_scheduling_tabulation
from greedy_interval_scheduling import (greedy_earliest_finish_time, greedy_highest_profit_first,
                                          greedy_best_ratio, demo_greedy_suboptimal_case)
import random

# Panel 1: match-rate bar chart (from the 500-trial experiment)
random.seed(7)
n_trials = 500
wins = {"Earliest finish time": 0, "Highest profit first": 0, "Best profit/duration ratio": 0}
for _ in range(n_trials):
    n = random.randint(3, 10)
    jobs = []
    for _ in range(n):
        s = random.randint(0, 15)
        e = random.randint(s + 1, 20)
        p = random.randint(1, 50)
        jobs.append((s, e, p))
    dp_profit, _ = weighted_job_scheduling_tabulation(jobs)
    gft, _ = greedy_earliest_finish_time(jobs)
    ghp, _ = greedy_highest_profit_first(jobs)
    gbr, _ = greedy_best_ratio(jobs)
    wins["Earliest finish time"] += (gft == dp_profit)
    wins["Highest profit first"] += (ghp == dp_profit)
    wins["Best profit/duration ratio"] += (gbr == dp_profit)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
labels = list(wins.keys())
pcts = [100 * v / n_trials for v in wins.values()]
bars = ax.bar(labels, pcts, color=["tab:red", "tab:orange", "tab:blue"])
ax.axhline(100, color="green", linestyle="--", alpha=0.5, label="DP optimum (always 100%)")
ax.set_ylabel("% of 500 random instances matching DP optimum")
ax.set_title("Greedy Heuristics vs DP Optimum\n(Weighted Interval Scheduling)")
ax.set_ylim(0, 105)
for b, p in zip(bars, pcts):
    ax.annotate(f"{p:.1f}%", (b.get_x() + b.get_width()/2, p), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=9)
ax.legend(fontsize=8)
ax.tick_params(axis='x', labelrotation=15)

# Panel 2: the constructed counter-example
jobs = [(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1), (1, 6, 100)]
greedy_profit, _ = greedy_earliest_finish_time(jobs)
dp_profit, _ = weighted_job_scheduling_tabulation(jobs)
ax2 = axes[1]
bars2 = ax2.bar(["Greedy\n(earliest finish time)", "DP\n(exact optimum)"], [greedy_profit, dp_profit],
                 color=["tab:red", "tab:green"])
ax2.set_ylabel("Total profit achieved")
ax2.set_title("Constructed Counter-Example\n(5 tiny jobs vs 1 high-value job)")
for b, v in zip(bars2, [greedy_profit, dp_profit]):
    ax2.annotate(str(v), (b.get_x() + b.get_width()/2, v), textcoords="offset points",
                 xytext=(0, 5), ha="center", fontsize=10, fontweight="bold")
ax2.annotate(f"Greedy achieves only {greedy_profit/dp_profit*100:.0f}% of optimal",
             xy=(0.5, 0.5), xycoords="axes fraction", ha="center", fontsize=9, color="darkred")

plt.tight_layout()
plt.savefig("fig_dp_vs_greedy_comparison.png", dpi=150)
print("Saved fig_dp_vs_greedy_comparison.png")
print("Match rates:", {k: f"{v}/{n_trials}" for k, v in wins.items()})