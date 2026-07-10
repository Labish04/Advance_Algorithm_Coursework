"""
Generates the labelled graphs required by Task 1's Analysis section:
    fig1_insert_time.png   - insert time vs n, log-log, all structures
    fig2_search_time.png   - search time vs n, log-log, BST/AVL/Hash
    fig3_height_growth.png - tree height vs n: BST (worst case) vs AVL (guaranteed log n)
    fig4_constant_factor.png - bar chart isolating the "hidden constant" AVL pays for balancing
"""
import csv
import math
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("Task1Results.csv")))
for r in rows:
    for k in ("n",):
        r[k] = int(r[k])
    for k in ("insert", "search", "delete"):
        r[k] = float(r[k]) if r[k] not in (None, "") else None
    r["height"] = int(r["height"]) if r["height"] not in (None, "") else None

ns = sorted(set(r["n"] for r in rows))


def series(structure, order, field):
    return [next(r[field] for r in rows if r["structure"] == structure and r["order"] == order and r["n"] == n)
            for n in ns]


# ---------------------------------------------------------------------------
# Fig 1: Insert time vs n (log-log), random-order input
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(ns, series("BST", "random", "insert"), "o-", label="BST (random input)")
plt.plot(ns, series("AVL", "random", "insert"), "s-", label="AVL (random input)")
plt.plot(ns, series("BST", "sorted", "insert"), "o--", color="red", label="BST (SORTED input, worst case)")
plt.plot(ns, series("AVL", "sorted", "insert"), "s--", color="green", label="AVL (sorted input)")
plt.plot(ns, series("HashTable", "random", "insert"), "^-", label="Hash Table")
plt.plot(ns, series("MinHeap(insert)", "random", "insert"), "d-", label="Min-Heap (n inserts)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("n (number of cities)")
plt.ylabel("Wall-clock time (seconds, median of 5 trials)")
plt.title("Insert Time vs n (log-log)\nBST degenerates to O(n) per op on sorted input; AVL stays O(log n)")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_insert_time.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Fig 2: Search time vs n (log-log)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(ns, series("BST", "random", "search"), "o-", label="BST (random input)")
plt.plot(ns, series("AVL", "random", "search"), "s-", label="AVL (random input)")
plt.plot(ns, series("BST", "sorted", "search"), "o--", color="red", label="BST (sorted input, O(n) search)")
plt.plot(ns, series("HashTable", "random", "search"), "^-", label="Hash Table (O(1) avg)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("n (number of cities)")
plt.ylabel("Wall-clock time for 100 search probes (seconds)")
plt.title("Search Time vs n (log-log)\nHash Table stays flat; degenerate BST grows linearly")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_search_time.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Fig 3: Height growth - BST worst case vs AVL guaranteed bound
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(ns, series("BST", "sorted", "height"), "o--", color="red", label="BST height (sorted input) — O(n)")
plt.plot(ns, series("BST", "random", "height"), "o-", color="orange", label="BST height (random input)")
plt.plot(ns, series("AVL", "sorted", "height"), "s--", color="green", label="AVL height (sorted input)")
plt.plot(ns, series("AVL", "random", "height"), "s-", color="darkgreen", label="AVL height (random input)")
theoretical_log2 = [math.log2(n) for n in ns]
plt.plot(ns, theoretical_log2, "k:", label="log2(n) reference line")
plt.xscale("log")
plt.xlabel("n (number of cities)")
plt.ylabel("Tree height")
plt.title("Tree Height vs n\nAVL guarantees O(log n) regardless of input order; BST does not")
plt.legend(fontsize=8)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_height_growth.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Fig 4: "Hidden constant" - AVL vs BST insert time on RANDOM (average-case) input
# Both are O(log n) on random input, yet AVL is measurably slower per-op due to
# rotation bookkeeping. This directly targets the brief's "acknowledge the
# constant factor hidden by Big-O" requirement.
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
width = 0.35
x = range(len(ns))
bst_vals = series("BST", "random", "insert")
avl_vals = series("AVL", "random", "insert")
plt.bar([i - width / 2 for i in x], bst_vals, width, label="BST insert (random)")
plt.bar([i + width / 2 for i in x], avl_vals, width, label="AVL insert (random)")
plt.xticks(list(x), [str(n) for n in ns])
plt.yscale("log")
plt.xlabel("n (number of cities)")
plt.ylabel("Wall-clock insert time (seconds)")
ratio = [a / b for a, b in zip(avl_vals, bst_vals)]
plt.title("BST vs AVL Insert Time on Random Input\n"
          f"Both are Theta(log n) asymptotically, but AVL is ~{sum(ratio)/len(ratio):.1f}x slower "
          "per operation\n(rotation + height bookkeeping = larger hidden constant)")
plt.legend(fontsize=8)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig4_constant_factor.png", dpi=150)
plt.close()

print("Saved fig1_insert_time.png, fig2_search_time.png, fig3_height_growth.png, fig4_constant_factor.png")
for n, r, a in zip(ns, bst_vals, avl_vals):
    pass
print("AVL/BST insert-time ratio per n (random input):", [f"{n}: {r:.2f}x" for n, r in zip(ns, ratio)])