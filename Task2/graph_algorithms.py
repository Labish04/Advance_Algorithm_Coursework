"""
Task 2: Graph Algorithms and Pathfinding for a Transportation Network
=======================================================================
Models a city transportation network as a weighted directed graph using an
adjacency list (justification in TASK2_ANALYSIS.md, Section 2.1).

Implements:
    - Dijkstra's algorithm     (single-source shortest path, non-negative weights)
    - Prim's algorithm         (Minimum Spanning Tree, undirected view of the graph)
    - Bellman-Ford algorithm   (handles negative weights, detects negative cycles)

Each algorithm returns not just the final result but also the full step-by-step
trace needed for the required execution visualisations.
"""
import heapq
from collections import defaultdict


class Graph:
    """Weighted directed graph, adjacency-list representation.

    Adjacency list chosen (over adjacency matrix) because real transportation
    networks are sparse (a city connects to a handful of neighbours, not all
    other cities): O(V+E) space and O(1) average edge iteration per vertex,
    versus O(V^2) for a matrix regardless of how sparse the graph actually is.
    See TASK2_ANALYSIS.md Section 2.1 for the full justification with numbers.
    """

    def __init__(self, directed=True):
        self.directed = directed
        self.adj = defaultdict(list)   # vertex -> list[(neighbour, weight)]
        self.vertices = set()

    def add_vertex(self, v):
        self.vertices.add(v)
        _ = self.adj[v]  # ensure key exists even with no out-edges

    def add_edge(self, u, v, w):
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append((v, w))
        if not self.directed:
            self.adj[v].append((u, w))

    def num_vertices(self):
        return len(self.vertices)

    def num_edges(self):
        return sum(len(es) for es in self.adj.values())

    def undirected_edge_view(self):
        """Returns a deduplicated list of (u, v, w) treating the graph as
        undirected — required for MST algorithms (Prim/Kruskal are defined
        on undirected graphs). Keeps the smaller-weight copy if both
        directions exist with different weights."""
        seen = {}
        for u in self.adj:
            for v, w in self.adj[u]:
                key = frozenset((u, v))
                if key not in seen or w < seen[key][2]:
                    seen[key] = (u, v, w)
        return list(seen.values())


# ---------------------------------------------------------------------------
# DIJKSTRA — single-source shortest path, non-negative weights only
# ---------------------------------------------------------------------------
def dijkstra(graph: Graph, source):
    """Returns (distances, predecessors, trace).
    trace is a list of steps: each step records the vertex finalised and the
    distance table snapshot at that point, for step-by-step visualisation.
    Time complexity: O((V + E) log V) with a binary heap.
    """
    dist = {v: float("inf") for v in graph.vertices}
    prev = {v: None for v in graph.vertices}
    dist[source] = 0
    visited = set()
    pq = [(0, source)]
    trace = []

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        trace.append({"finalised": u, "distance": d, "dist_snapshot": dict(dist)})

        for v, w in graph.adj[u]:
            if w < 0:
                raise ValueError("Dijkstra does not support negative edge weights")
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    return dist, prev, trace


def reconstruct_path(prev, source, target):
    if prev.get(target) is None and target != source:
        return None
    path = [target]
    while path[-1] != source:
        p = prev[path[-1]]
        if p is None:
            return None
        path.append(p)
    return list(reversed(path))


# ---------------------------------------------------------------------------
# PRIM'S ALGORITHM — Minimum Spanning Tree (undirected)
# ---------------------------------------------------------------------------
def prim_mst(graph: Graph, start=None):
    """Returns (mst_edges, total_weight, trace).
    Operates on the undirected view of the graph (MST is only defined for
    undirected graphs). Time complexity: O((V + E) log V) with a binary heap
    and adjacency list.
    """
    # Build undirected adjacency for Prim's traversal
    und_adj = defaultdict(list)
    for u, v, w in graph.undirected_edge_view():
        und_adj[u].append((v, w))
        und_adj[v].append((u, w))

    vertices = list(graph.vertices)
    if not vertices:
        return [], 0, []
    start = start or vertices[0]

    visited = {start}
    edges_heap = [(w, start, v) for v, w in und_adj[start]]
    heapq.heapify(edges_heap)
    mst_edges = []
    total_weight = 0
    trace = [{"visited": set(visited), "mst_edges": [], "total_weight": 0}]

    while edges_heap and len(visited) < len(vertices):
        w, u, v = heapq.heappop(edges_heap)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        total_weight += w
        for nxt, w2 in und_adj[v]:
            if nxt not in visited:
                heapq.heappush(edges_heap, (w2, v, nxt))
        trace.append({"visited": set(visited), "mst_edges": list(mst_edges), "total_weight": total_weight})

    connected = len(visited) == len(vertices)
    return mst_edges, total_weight, trace, connected


# ---------------------------------------------------------------------------
# BELLMAN-FORD — handles negative weights, detects negative cycles
# ---------------------------------------------------------------------------
def bellman_ford(graph: Graph, source):
    """Returns (distances, predecessors, has_negative_cycle, trace).
    Time complexity: O(V*E) — relaxes every edge V-1 times, then does one
    more pass to detect a negative cycle.
    """
    dist = {v: float("inf") for v in graph.vertices}
    prev = {v: None for v in graph.vertices}
    dist[source] = 0
    edges = [(u, v, w) for u in graph.adj for v, w in graph.adj[u]]
    trace = []

    V = len(graph.vertices)
    for i in range(V - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        trace.append({"iteration": i, "dist_snapshot": dict(dist)})
        if not updated:
            break  # early exit: converged before V-1 passes (common in practice)

    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, prev, has_negative_cycle, trace


# ---------------------------------------------------------------------------
# Synthetic transportation-network generator
# ---------------------------------------------------------------------------
import random


def generate_random_graph(n_vertices, density="sparse", directed=True, seed=0, allow_negative=False):
    """density: 'sparse' -> ~ n_vertices edges per vertex is small (E ~ 2V-4V);
                'dense'  -> E close to V*(V-1) (near-complete graph)."""
    rng = random.Random(seed)
    g = Graph(directed=directed)
    vertices = [f"V{i}" for i in range(n_vertices)]
    for v in vertices:
        g.add_vertex(v)

    if density == "sparse":
        target_edges = min(n_vertices * 3, n_vertices * (n_vertices - 1))
    else:  # dense
        target_edges = int(0.6 * n_vertices * (n_vertices - 1))

    # Build a canonical edge map first, keyed so that (u,v) and (v,u) can never
    # be inserted as two distinct entries with different weights -- undirected
    # graphs key on frozenset({u,v}); directed graphs key on the ordered pair
    # (a directed graph CAN legitimately have different weights for u->v and
    # v->u, so those remain distinct).
    edge_weights = {}

    def edge_key(u, v):
        return frozenset((u, v)) if not directed else (u, v)

    # Guarantee connectivity via a random chain through all vertices first.
    chain = vertices[:]
    rng.shuffle(chain)
    for i in range(len(chain) - 1):
        w = rng.randint(1, 20)
        edge_weights[edge_key(chain[i], chain[i + 1])] = (chain[i], chain[i + 1], w)

    attempts = 0
    while len(edge_weights) < target_edges and attempts < target_edges * 20:
        attempts += 1
        u, v = rng.sample(vertices, 2)
        key = edge_key(u, v)
        if key in edge_weights:
            continue
        w = rng.randint(-5, 20) if allow_negative else rng.randint(1, 20)
        edge_weights[key] = (u, v, w)

    g2 = Graph(directed=directed)
    for v in vertices:
        g2.add_vertex(v)
    for (u, v, w) in edge_weights.values():
        g2.add_edge(u, v, w)

    return g2


if __name__ == "__main__":
    # Small worked example
    g = Graph(directed=True)
    edges = [("A", "B", 4), ("A", "C", 1), ("C", "B", 2), ("B", "D", 5), ("C", "D", 8), ("D", "E", 3)]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    dist, prev, trace = dijkstra(g, "A")
    print("Dijkstra distances from A:", dist)
    print("Path A->E:", reconstruct_path(prev, "A", "E"))

    mst_edges, total_w, mst_trace, connected = prim_mst(g)
    print("Prim MST edges:", mst_edges, "total weight:", total_w, "connected:", connected)

    dist_bf, prev_bf, neg_cycle, bf_trace = bellman_ford(g, "A")
    print("Bellman-Ford distances from A:", dist_bf, "negative cycle:", neg_cycle)

    # negative-weight example
    g_neg = Graph(directed=True)
    for u, v, w in [("A", "B", 4), ("A", "C", 5), ("B", "C", -3), ("C", "D", 2), ("B", "D", 6)]:
        g_neg.add_edge(u, v, w)
    dist_bf2, prev_bf2, neg2, _ = bellman_ford(g_neg, "A")
    print("\nBellman-Ford with negative edge B->C=-3:", dist_bf2, "negative cycle:", neg2)

    # negative cycle example
    g_cycle = Graph(directed=True)
    for u, v, w in [("A", "B", 1), ("B", "C", -1), ("C", "A", -1)]:
        g_cycle.add_edge(u, v, w)
    _, _, neg3, _ = bellman_ford(g_cycle, "A")
    print("Negative cycle detected (should be True):", neg3)