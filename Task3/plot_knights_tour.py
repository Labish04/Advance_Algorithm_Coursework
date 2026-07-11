import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("knights_tour_results.csv")))
ns = [int(r["n"]) for r in rows]
naive_calls = [int(r["naive_calls"]) for r in rows]
warn_calls = [int(r["warnsdorff_calls"]) for r in rows]
naive_hit_limit = [r["naive_hit_limit"] == "True" for r in rows]

fig, ax1 = plt.subplots(figsize=(8, 5.5))
bars1 = ax1.bar([n - 0.2 for n in ns], naive_calls, width=0.4, label="Naive backtracking (recursive calls)",
                color="tab:red")
bars2 = ax1.bar([n + 0.2 for n in ns], warn_calls, width=0.4, label="Warnsdorff heuristic (recursive calls)",
                color="tab:green")
ax1.set_yscale("log")
ax1.set_xlabel("Board size n (n x n Knight's Tour)")
ax1.set_ylabel("Recursive calls made (log scale)")
ax1.set_xticks(ns)
ax1.legend(loc="upper left")
ax1.grid(True, axis="y", alpha=0.3)

for n, c, hit in zip(ns, naive_calls, naive_hit_limit):
    label = f"{c:,}" + (" (capped!)" if hit else "")
    ax1.annotate(label, (n - 0.2, c), textcoords="offset points", xytext=(0, 5),
                 ha="center", fontsize=8, color="darkred")
for n, c in zip(ns, warn_calls):
    ax1.annotate(f"{c}", (n + 0.2, c), textcoords="offset points", xytext=(0, 5),
                 ha="center", fontsize=8, color="darkgreen")

plt.title("Knight's Tour: Search Space Explored, Naive vs Warnsdorff-Pruned Backtracking\n"
          "Warnsdorff makes exactly n² calls (zero backtracking); naive search grows combinatorially "
          "and is capped at 2,000,000 calls for n≥7")
plt.tight_layout()
plt.savefig("fig_knights_tour_search_space.png", dpi=150)
plt.close()
print("Saved fig_knights_tour_search_space.png")

# growth factor
print("Naive call-count growth factor between consecutive n:")
for i in range(1, len(ns)):
    if not naive_hit_limit[i] and not naive_hit_limit[i - 1]:
        print(f"  n={ns[i-1]}->{ns[i]}: {naive_calls[i]/naive_calls[i-1]:.1f}x")