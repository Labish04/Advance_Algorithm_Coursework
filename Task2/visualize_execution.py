"""
Produces step-by-step visualisations of algorithm execution, as required by
Task 2's brief ("Provide visualisations of algorithm execution, e.g.
step-by-step shortest path tree or MST construction").

Outputs:
    fig3_dijkstra_steps.png  - 6-panel snapshot of Dijkstra building its SPT
    fig4_prim_mst_steps.png  - 6-panel snapshot of Prim building the MST
    fig5_bellman_ford_convergence.png - distance-table convergence per iteration
"""
import matplotlib.pyplot as plt
import networkx as nx
from graph_algorithms import Graph, dijkstra, prim_mst, bellman_ford

# A small, hand-crafted example graph so the visualisation is legible
edges = [
    ("A", "B", 4), ("A", "C", 1), ("C", "B", 2), ("B", "D", 5),
    ("C", "D", 8), ("D", "E", 3), ("C", "E", 10), ("B", "E", 6),
]

g = Graph(directed=True)
for u, v, w in edges:
    g.add_edge(u, v, w)

G = nx.DiGraph()
for u, v, w in edges:
    G.add_edge(u, v, weight=w)
pos = nx.spring_layout(G, seed=42)


# ---------------------------------------------------------------------------
# Fig 3: Dijkstra step-by-step
# ---------------------------------------------------------------------------
dist, prev, trace = dijkstra(g, "A")

n_steps = len(trace)
fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
if n_steps == 1:
    axes = [axes]

for i, (ax, step) in enumerate(zip(axes, trace)):
    finalised_so_far = [t["finalised"] for t in trace[: i + 1]]
    node_colors = ["tab:green" if n in finalised_so_far else "lightgray" for n in G.nodes()]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            node_size=600, font_weight="bold", arrows=True)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)
    dist_str = ", ".join(f"{k}:{v if v != float('inf') else 'inf'}" for k, v in step["dist_snapshot"].items())
    ax.set_title(f"Step {i+1}: finalise {step['finalised']} (d={step['distance']})\n{dist_str}", fontsize=8)

fig.suptitle("Dijkstra's Algorithm: Step-by-Step Shortest-Path Tree Construction from A\n"
             "(green = vertex finalised; labels show live distance table)", fontsize=11)
plt.tight_layout()
plt.savefig("fig3_dijkstra_steps.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Fig 4: Prim's MST step-by-step (undirected view)
# ---------------------------------------------------------------------------
g_und = Graph(directed=False)
for u, v, w in edges:
    g_und.add_edge(u, v, w)
G_und = nx.Graph()
for u, v, w in edges:
    G_und.add_edge(u, v, weight=w)
pos_und = nx.spring_layout(G_und, seed=42)

mst_edges, total_w, mst_trace, connected = prim_mst(g_und)

n_steps = len(mst_trace)
fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
if n_steps == 1:
    axes = [axes]

for i, (ax, step) in enumerate(zip(axes, mst_trace)):
    node_colors = ["tab:green" if n in step["visited"] else "lightgray" for n in G_und.nodes()]
    nx.draw(G_und, pos_und, ax=ax, with_labels=True, node_color=node_colors,
            node_size=600, font_weight="bold")
    mst_edge_set = {frozenset((u, v)) for u, v, w in step["mst_edges"]}
    edge_colors = ["tab:red" if frozenset((u, v)) in mst_edge_set else "lightgray"
                   for u, v in G_und.edges()]
    edge_widths = [3 if frozenset((u, v)) in mst_edge_set else 1 for u, v in G_und.edges()]
    nx.draw_networkx_edges(G_und, pos_und, ax=ax, edge_color=edge_colors, width=edge_widths)
    edge_labels = nx.get_edge_attributes(G_und, "weight")
    nx.draw_networkx_edge_labels(G_und, pos_und, edge_labels=edge_labels, ax=ax, font_size=7)
    ax.set_title(f"Step {i}: {len(step['mst_edges'])} edges, weight={step['total_weight']}", fontsize=9)

fig.suptitle("Prim's Algorithm: Step-by-Step Minimum Spanning Tree Construction\n"
             f"(green = vertex added; red = MST edge; final MST weight = {total_w})", fontsize=11)
plt.tight_layout()
plt.savefig("fig4_prim_mst_steps.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Fig 5: Bellman-Ford convergence (distances per iteration, on a graph with
# a negative edge so the "handles negative weights" behaviour is visible)
# ---------------------------------------------------------------------------
neg_edges = [("A", "B", 4), ("A", "C", 5), ("B", "C", -3), ("C", "D", 2), ("B", "D", 6), ("D", "E", 1)]
g_neg = Graph(directed=True)
for u, v, w in neg_edges:
    g_neg.add_edge(u, v, w)
dist_bf, prev_bf, has_cycle, bf_trace = bellman_ford(g_neg, "A")

vertices_order = sorted(g_neg.vertices)
plt.figure(figsize=(7, 5))
for v in vertices_order:
    ys = [step["dist_snapshot"][v] for step in bf_trace]
    ys = [y if y != float("inf") else None for y in ys]
    xs = list(range(1, len(bf_trace) + 1))
    plt.plot(xs, ys, "o-", label=f"dist[{v}]")
plt.xlabel("Bellman-Ford iteration (edge-relaxation pass)")
plt.ylabel("Shortest distance from A (so far)")
plt.title(f"Bellman-Ford Convergence with a Negative Edge (B->C = -3)\n"
          f"Negative cycle detected: {has_cycle} — distances stabilise after {len(bf_trace)} passes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig5_bellman_ford_convergence.png", dpi=150)
plt.close()

print("Saved fig3_dijkstra_steps.png, fig4_prim_mst_steps.png, fig5_bellman_ford_convergence.png")
print("Dijkstra final distances:", dist)
print("Prim MST edges:", mst_edges, "total weight:", total_w)
print("Bellman-Ford final distances (with negative edge):", dist_bf)