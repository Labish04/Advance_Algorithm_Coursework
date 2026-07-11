"""
Generates comparison graphs for Task 2's Analysis section:
    fig1_sparse_vs_dense.png  - all 3 algorithms, sparse vs dense, vs n
    fig2_bellman_ford_scaling.png - BF time vs E, showing O(V*E) growth
"""
import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("Task2Results.csv")))
for r in rows:
    r["n"] = int(r["n"])
    r["E_directed"] = int(r["E_directed"])
    r["E_undirected"] = int(r["E_undirected"])
    for k in ("dijkstra", "bellman_ford", "prim"):
        r[k] = float(r[k])

ns = sorted(set(r["n"] for r in rows))


def series(density, field):
    return [next(r[field] for r in rows if r["density"] == density and r["n"] == n) for n in ns]


# ---------------------------------------------------------------------------
# Fig 1: All three algorithms, sparse vs dense
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, density in zip(axes, ["sparse", "dense"]):
    ax.plot(ns, series(density, "dijkstra"), "o-", label="Dijkstra O((V+E)logV)")
    ax.plot(ns, series(density, "bellman_ford"), "s-", label="Bellman-Ford O(V*E)")
    ax.plot(ns, series(density, "prim"), "^-", label="Prim O((V+E)logV)")
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel("Wall-clock time (s, median of 5 trials)")
    ax.set_title(f"{density.capitalize()} Graphs")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Algorithm Runtime vs Graph Size: Sparse vs Dense\n"
             "Bellman-Ford's O(V*E) cost is hidden on sparse graphs but dominates on dense ones")
plt.tight_layout()
plt.savefig("fig1_sparse_vs_dense.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Fig 2: Bellman-Ford time vs number of edges E (should be ~linear in E for fixed V-1 passes)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
all_E = [r["E_directed"] for r in rows]
all_BF = [r["bellman_ford"] for r in rows]
all_Dij = [r["dijkstra"] for r in rows]
plt.scatter(all_E, all_BF, label="Bellman-Ford", color="tab:orange")
plt.scatter(all_E, all_Dij, label="Dijkstra", color="tab:blue")
plt.xlabel("Number of edges E")
plt.ylabel("Wall-clock time (s)")
plt.xscale("log")
plt.yscale("log")
plt.title("Time vs Edge Count E\nBellman-Ford (O(V*E)) grows faster with E than Dijkstra (O((V+E)logV))")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_bellman_ford_scaling.png", dpi=150)
plt.close()

print("Saved fig1_sparse_vs_dense.png, fig2_bellman_ford_scaling.png")

# Print the key ratio for the write-up
dense_rows = [r for r in rows if r["density"] == "dense"]
for r in dense_rows:
    print(f"n={r['n']}: BF/Dijkstra ratio on DENSE graph = {r['bellman_ford']/r['dijkstra']:.2f}x")