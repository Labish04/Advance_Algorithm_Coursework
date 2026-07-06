"""
Empirical benchmarking for Task 1.

Measures wall-clock time for INSERT, SEARCH, and DELETE across
n = 100, 1,000, 10,000 for:
    - BST
    - AVL Tree
    - Min-Heap        (insert + extract-min, as it has no keyed search/delete)
    - Hash Table      (insert + search + delete by key)

Two input distributions are tested:
    - RANDOM order    (average-case for BST)
    - SORTED order    (worst-case / adversarial for BST -> degenerates to O(n))

Results are written to results.csv and plotted to png files (log-scale) for
direct comparison against theoretical Big-O predictions.
"""
import time
import random
import csv
import statistics as stats
from data_structures import BST, AVLTree, MinHeap, HashTable, City

N_VALUES = [100, 1_000, 10_000]
N_SEARCH_SAMPLES = 100   # number of search probes per trial (averaged)
N_TRIALS = 5             # repeat each measurement and take the median (reduces noise)


def make_cities(n, order="random", seed=0):
    rng = random.Random(seed)
    cities = [City(f"c{i}", 0.0, 0.0, 1, distance=float(i)) for i in range(n)]
    if order == "random":
        rng.shuffle(cities)
    # "sorted" order left as-is (ascending distance) -> BST worst case
    return cities


def time_it(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def median_of_trials(fn, trials=N_TRIALS):
    times = [time_it(fn) for _ in range(trials)]
    return stats.median(times)


def bench_bst(cities, search_keys):
    def build():
        t = BST()
        for c in cities:
            t.insert(c)
        return t

    t_insert = median_of_trials(lambda: build())
    tree = build()
    t_search = median_of_trials(lambda: [tree.search(k) for k in search_keys])
    # deletion: delete first 100 (or all if n<100) keys, rebuild each trial (delete mutates)
    del_keys = [c.distance for c in cities[: min(100, len(cities))]]

    def do_deletes():
        tr = build()
        for k in del_keys:
            tr.delete(k)

    t_delete = median_of_trials(do_deletes)
    return t_insert, t_search, t_delete, tree.height()


def bench_avl(cities, search_keys):
    def build():
        t = AVLTree()
        for c in cities:
            t.insert(c)
        return t

    t_insert = median_of_trials(lambda: build())
    tree = build()
    t_search = median_of_trials(lambda: [tree.search(k) for k in search_keys])
    del_keys = [c.distance for c in cities[: min(100, len(cities))]]

    def do_deletes():
        tr = build()
        for k in del_keys:
            tr.delete(k)

    t_delete = median_of_trials(do_deletes)
    return t_insert, t_search, t_delete, tree.height()


def bench_heap(cities):
    def build_via_inserts():
        h = MinHeap()
        for c in cities:
            h.insert(c)
        return h

    t_insert = median_of_trials(build_via_inserts)

    def build_via_heapify():
        return MinHeap.build_heap(cities)

    t_build_heapify = median_of_trials(build_via_heapify)

    def do_extracts():
        h = build_via_heapify()
        for _ in range(min(100, len(cities))):
            h.extract_min()

    t_extract = median_of_trials(do_extracts)
    return t_insert, t_build_heapify, t_extract


def bench_hash(cities):
    def build():
        h = HashTable(capacity=16)
        for c in cities:
            h.insert(c.name, c)
        return h

    t_insert = median_of_trials(build)
    table = build()
    search_keys = [c.name for c in random.sample(cities, min(N_SEARCH_SAMPLES, len(cities)))]
    t_search = median_of_trials(lambda: [table.search(k) for k in search_keys])
    del_keys = [c.name for c in cities[: min(100, len(cities))]]

    def do_deletes():
        h = build()
        for k in del_keys:
            h.delete(k)

    t_delete = median_of_trials(do_deletes)
    return t_insert, t_search, t_delete


def run_all():
    rows = []
    for n in N_VALUES:
        # ---- RANDOM ORDER (average case) ----
        cities_rand = make_cities(n, order="random", seed=1)
        search_keys_rand = [c.distance for c in random.Random(2).sample(cities_rand, min(N_SEARCH_SAMPLES, n))]

        bi, bs, bd, bh = bench_bst(cities_rand, search_keys_rand)
        ai, as_, ad, ah = bench_avl(cities_rand, search_keys_rand)
        hi, hb, he = bench_heap(cities_rand)
        hti, hts, htd = bench_hash(cities_rand)

        rows.append(dict(n=n, order="random", structure="BST", insert=bi, search=bs, delete=bd, height=bh))
        rows.append(dict(n=n, order="random", structure="AVL", insert=ai, search=as_, delete=ad, height=ah))
        rows.append(dict(n=n, order="random", structure="MinHeap(insert)", insert=hi, search=None, delete=None, height=None))
        rows.append(dict(n=n, order="random", structure="MinHeap(heapify+extract)", insert=hb, search=None, delete=he, height=None))
        rows.append(dict(n=n, order="random", structure="HashTable", insert=hti, search=hts, delete=htd, height=None))

        # ---- SORTED ORDER (BST worst case) ----
        cities_sorted = make_cities(n, order="sorted", seed=1)
        search_keys_sorted = [float(k) for k in random.Random(2).sample(range(n), min(N_SEARCH_SAMPLES, n))]

        bi2, bs2, bd2, bh2 = bench_bst(cities_sorted, search_keys_sorted)
        ai2, as2, ad2, ah2 = bench_avl(cities_sorted, search_keys_sorted)

        rows.append(dict(n=n, order="sorted", structure="BST", insert=bi2, search=bs2, delete=bd2, height=bh2))
        rows.append(dict(n=n, order="sorted", structure="AVL", insert=ai2, search=as2, delete=ad2, height=ah2))

        print(f"n={n} done "
              f"(BST height rand={bh}/sorted={bh2}, AVL height rand={ah}/sorted={ah2})")

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "order", "structure", "insert", "search", "delete", "height"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results.csv")
    return rows


if __name__ == "__main__":
    run_all()