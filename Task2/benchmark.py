"""
Benchmarks Dijkstra, Prim, and Bellman-Ford across:
    - varying n (number of vertices): 20, 50, 100, 200
    - sparse vs dense graphs
to empirically test the theoretical predictions:
    Dijkstra (heap):    O((V+E) log V)
    Prim (heap):        O((V+E) log V)
    Bellman-Ford:       O(V*E)   <- should scale far worse on dense graphs
"""
import time
import statistics as stats
import csv
from graph_algorithms import Graph, dijkstra, prim_mst, bellman_ford, generate_random_graph

N_VALUES = [20, 50, 100, 200]
N_TRIALS = 5


def median_time(fn, trials=N_TRIALS):
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return stats.median(times)


def run_all():
    rows = []
    for n in N_VALUES:
        for density in ["sparse", "dense"]:
            g = generate_random_graph(n, density=density, directed=True, seed=1)
            source = next(iter(g.vertices))
            E = g.num_edges()

            t_dij = median_time(lambda: dijkstra(g, source))
            t_bf = median_time(lambda: bellman_ford(g, source))

            # Prim needs an undirected graph
            g_undirected = generate_random_graph(n, density=density, directed=False, seed=1)
            E_und = g_undirected.num_edges()
            t_prim = median_time(lambda: prim_mst(g_undirected))

            rows.append(dict(n=n, density=density, E_directed=E, E_undirected=E_und,
                              dijkstra=t_dij, bellman_ford=t_bf, prim=t_prim))
            print(f"n={n:4d} density={density:6s}  E(dir)={E:6d}  "
                  f"Dijkstra={t_dij:.6f}s  BellmanFord={t_bf:.6f}s  Prim={t_prim:.6f}s")

    with open("Task2Results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "density", "E_directed", "E_undirected",
                                                "dijkstra", "bellman_ford", "prim"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved Task2Results.csv")
    return rows


if __name__ == "__main__":
    run_all()