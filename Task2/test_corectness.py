"""
Correctness tests for Task 2 — cross-validated against networkx reference
implementations to independently confirm our results are correct.
"""
import networkx as nx
from graph_algorithms import Graph, dijkstra, prim_mst, bellman_ford, generate_random_graph


def to_networkx(g: Graph):
    G = nx.DiGraph() if g.directed else nx.Graph()
    G.add_nodes_from(g.vertices)
    for u in g.adj:
        for v, w in g.adj[u]:
            G.add_edge(u, v, weight=w)
    return G


def test_dijkstra_matches_networkx(trials=10):
    for seed in range(trials):
        g = generate_random_graph(30, density="sparse", directed=True, seed=seed)
        G = to_networkx(g)
        source = next(iter(g.vertices))

        our_dist, _, _ = dijkstra(g, source)
        nx_dist = nx.single_source_dijkstra_path_length(G, source)

        for v in g.vertices:
            expected = nx_dist.get(v, float("inf"))
            if our_dist[v] == float("inf") and expected == float("inf"):
                continue
            assert abs(our_dist[v] - expected) < 1e-9, f"Mismatch at {v}: {our_dist[v]} vs {expected}"
    print(f"[PASS] Dijkstra matches networkx on {trials} random sparse graphs (n=30)")


def test_prim_matches_networkx(trials=10):
    for seed in range(trials):
        g = generate_random_graph(20, density="sparse", directed=False, seed=seed)
        G = to_networkx(g)

        _, our_weight, _, connected = prim_mst(g)
        if not connected:
            continue
        nx_mst = nx.minimum_spanning_tree(G)
        nx_weight = sum(d["weight"] for _, _, d in nx_mst.edges(data=True))

        assert our_weight == nx_weight, f"MST weight mismatch: {our_weight} vs {nx_weight}"
    print(f"[PASS] Prim's MST weight matches networkx on {trials} random graphs (n=20)")


def test_bellman_ford_matches_networkx_negative_weights(trials=10):
    for seed in range(trials):
        g = generate_random_graph(15, density="sparse", directed=True, seed=seed, allow_negative=True)
        G = to_networkx(g)
        source = next(iter(g.vertices))

        our_dist, _, our_neg_cycle, _ = bellman_ford(g, source)

        try:
            nx_dist = nx.single_source_bellman_ford_path_length(G, source)
            nx_has_cycle = False
        except nx.NetworkXUnbounded:
            nx_has_cycle = True
            nx_dist = None

        assert our_neg_cycle == nx_has_cycle, f"Negative-cycle detection mismatch (seed={seed})"
        if not nx_has_cycle:
            for v in g.vertices:
                expected = nx_dist.get(v, float("inf"))
                if our_dist[v] == float("inf") and expected == float("inf"):
                    continue
                assert abs(our_dist[v] - expected) < 1e-9, f"Mismatch at {v}: {our_dist[v]} vs {expected}"
    print(f"[PASS] Bellman-Ford matches networkx (incl. negative-cycle detection) on {trials} graphs")


def test_bellman_ford_detects_known_negative_cycle():
    g = Graph(directed=True)
    for u, v, w in [("A", "B", 1), ("B", "C", -1), ("C", "A", -1)]:
        g.add_edge(u, v, w)
    _, _, has_cycle, _ = bellman_ford(g, "A")
    assert has_cycle
    print("[PASS] Bellman-Ford correctly detects a hand-crafted negative cycle")


def test_dijkstra_rejects_negative_edge():
    g = Graph(directed=True)
    g.add_edge("A", "B", -5)
    try:
        dijkstra(g, "A")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[PASS] Dijkstra correctly refuses negative edge weights")


if __name__ == "__main__":
    test_dijkstra_matches_networkx()
    test_prim_matches_networkx()
    test_bellman_ford_matches_networkx_negative_weights()
    test_bellman_ford_detects_known_negative_cycle()
    test_dijkstra_rejects_negative_edge()
    print("\nAll Task 2 correctness tests passed.")