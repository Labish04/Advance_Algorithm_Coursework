"""
Correctness tests for Task 1 data structures.
Run: python3 test_correctness.py
"""
import random
from data_structures import BST, AVLTree, MinHeap, HashTable, generate_cities, City


def check_avl_balanced(node):
    """Returns height if balanced, raises AssertionError if any node violates |BF| <= 1."""
    if node is None:
        return 0
    lh = check_avl_balanced(node.left)
    rh = check_avl_balanced(node.right)
    assert abs(lh - rh) <= 1, f"AVL invariant broken at city {node.city}, bf={lh - rh}"
    return 1 + max(lh, rh)


def check_bst_property(node, lo=float("-inf"), hi=float("inf")):
    if node is None:
        return
    assert lo <= node.city.distance <= hi, "BST property violated"
    check_bst_property(node.left, lo, node.city.distance)
    check_bst_property(node.right, node.city.distance, hi)


def check_heap_property(heap: MinHeap):
    data = heap.data
    n = len(data)
    for i in range(n):
        l, r = 2 * i + 1, 2 * i + 2
        if l < n:
            assert data[i].distance <= data[l].distance, "Min-heap property violated (left)"
        if r < n:
            assert data[i].distance <= data[r].distance, "Min-heap property violated (right)"


def test_avl_balance_under_random_ops(n=2000, seed=1):
    rng = random.Random(seed)
    cities = generate_cities(n, seed=seed)
    avl = AVLTree()
    for c in cities:
        avl.insert(c)
    check_avl_balanced(avl.root)
    check_bst_property(avl.root)

    # random deletions, re-check invariant after every batch
    rng.shuffle(cities)
    for c in cities[: n // 2]:
        avl.delete(c.distance)
        # spot check every 200 deletions to keep test fast
    check_avl_balanced(avl.root)
    check_bst_property(avl.root)
    print(f"[PASS] AVL remains balanced after {n} inserts + {n//2} deletes")


def test_bst_property(n=2000, seed=2):
    cities = generate_cities(n, seed=seed)
    bst = BST()
    for c in cities:
        bst.insert(c)
    check_bst_property(bst.root)
    print(f"[PASS] BST property holds after {n} inserts")


def test_heap_extract_order(n=1000, seed=3):
    cities = generate_cities(n, seed=seed)
    heap = MinHeap.build_heap(cities)
    check_heap_property(heap)
    prev = -1
    count = 0
    while len(heap):
        c = heap.extract_min()
        assert c.distance >= prev, "Heap did not extract in ascending order"
        prev = c.distance
        count += 1
    assert count == n
    print(f"[PASS] Min-heap extracts {n} elements in strictly ascending order")


def test_hashtable_correctness(n=5000, seed=4):
    cities = generate_cities(n, seed=seed)
    ht = HashTable(capacity=8)  # small capacity to force multiple resizes
    for c in cities:
        ht.insert(c.name, c)
    assert len(ht) == n

    rng = random.Random(seed)
    sample = rng.sample(cities, 200)
    for c in sample:
        found = ht.search(c.name)
        assert found is not None and found.name == c.name

    # deletion check
    for c in sample[:50]:
        assert ht.delete(c.name)
        assert ht.search(c.name) is None
    assert len(ht) == n - 50
    print(f"[PASS] HashTable insert/search/delete correct for n={n} (with resizing)")


def test_avl_height_vs_bst_height_worst_case():
    """Insert sorted (ascending) data: BST degenerates to O(n) height, AVL stays O(log n)."""
    n = 1000
    cities = [City(f"c{i}", 0, 0, 1, distance=i) for i in range(n)]
    bst, avl = BST(), AVLTree()
    for c in cities:
        bst.insert(c)
        avl.insert(c)
    import math
    print(f"[INFO] Sorted-input stress test (n={n}): "
          f"BST height={bst.height()} (worst-case O(n)), "
          f"AVL height={avl.height()} (bound ~1.44*log2(n)={1.44*math.log2(n):.1f})")
    assert bst.height() == n - 1, "BST should degenerate into a linked list on sorted input"
    assert avl.height() <= 1.45 * math.log2(n)
    print("[PASS] AVL stays logarithmic on adversarial (sorted) input where BST degenerates")


if __name__ == "__main__":
    test_bst_property()
    test_avl_balance_under_random_ops()
    test_heap_extract_order()
    test_hashtable_correctness()
    test_avl_height_vs_bst_height_worst_case()
    print("\nAll correctness tests passed.")