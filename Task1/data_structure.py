"""
Task 1: Advanced Data Structures for a Route Planning Application
====================================================================
Implements:
    - Binary Search Tree (BST)          - keyed on city distance-from-origin
    - AVL Tree (self-balancing BST)      - same key, guarantees O(log n) ops
    - Min-Heap (array-based)             - priority queue, "next nearest city"
    - Hash Table (separate chaining)     - O(1) average city lookup by name

Each city record stores: name, coordinates (lat, lon), population, and
distance (used as the ordering key for BST/AVL/Heap; hash table indexes by name).

Author: Labish Parajuli
"""

from dataclasses import dataclass
import random
import sys


sys.setrecursionlimit(20000)


@dataclass
class City:
    name: str
    lat: float
    lon: float
    population: int
    distance: float  # distance from a reference origin point (the ordering key)

    def __repr__(self):
        return f"City({self.name}, dist={self.distance:.2f}, pop={self.population})"


# ---------------------------------------------------------------------------
# 1. BINARY SEARCH TREE (unbalanced)
# ---------------------------------------------------------------------------
class BSTNode:
    __slots__ = ("city", "left", "right")

    def __init__(self, city: City):
        self.city = city
        self.left = None
        self.right = None


class BST:
    """Standard unbalanced BST keyed on City.distance."""

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    def insert(self, city: City):
        self._size += 1
        if self.root is None:
            self.root = BSTNode(city)
            return
        node = self.root
        while True:
            if city.distance < node.city.distance:
                if node.left is None:
                    node.left = BSTNode(city)
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = BSTNode(city)
                    return
                node = node.right

    def search(self, distance: float):
        node = self.root
        while node is not None:
            if distance == node.city.distance:
                return node.city
            node = node.left if distance < node.city.distance else node.right
        return None

    def delete(self, distance: float):
        self.root, deleted = self._delete(self.root, distance)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node, distance):
        if node is None:
            return node, False
        if distance < node.city.distance:
            node.left, deleted = self._delete(node.left, distance)
            return node, deleted
        elif distance > node.city.distance:
            node.right, deleted = self._delete(node.right, distance)
            return node, deleted
        else:
            # found node to delete
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True
            # two children: replace with inorder successor
            succ_parent = node
            succ = node.right
            while succ.left is not None:
                succ_parent = succ
                succ = succ.left
            node.city = succ.city
            if succ_parent.left is succ:
                succ_parent.left = succ.right
            else:
                succ_parent.right = succ.right
            return node, True

    def height(self):
        """Iterative (stack-based) height computation - avoids Python recursion-depth
        limits on a degenerate/sorted-input BST where height can reach O(n)."""
        if self.root is None:
            return -1
        max_h = -1
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            max_h = max(max_h, depth)
            if node.left:
                stack.append((node.left, depth + 1))
            if node.right:
                stack.append((node.right, depth + 1))
        return max_h


# ---------------------------------------------------------------------------
# 2. AVL TREE (self-balancing BST)
# ---------------------------------------------------------------------------
class AVLNode:
    __slots__ = ("city", "left", "right", "height")

    def __init__(self, city: City):
        self.city = city
        self.left = None
        self.right = None
        self.height = 1  # height of subtree rooted here (leaf = 1)


class AVLTree:
    """Self-balancing BST. Maintains |balance factor| <= 1 at every node."""

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    @staticmethod
    def _h(node):
        return node.height if node else 0

    @staticmethod
    def _bf(node):
        return AVLTree._h(node.left) - AVLTree._h(node.right) if node else 0

    @staticmethod
    def _update_height(node):
        node.height = 1 + max(AVLTree._h(node.left), AVLTree._h(node.right))

    @staticmethod
    def _rotate_right(y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        AVLTree._update_height(y)
        AVLTree._update_height(x)
        return x

    @staticmethod
    def _rotate_left(x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        AVLTree._update_height(x)
        AVLTree._update_height(y)
        return y

    def insert(self, city: City):
        self.root = self._insert(self.root, city)
        self._size += 1

    def _insert(self, node, city):
        if node is None:
            return AVLNode(city)
        if city.distance < node.city.distance:
            node.left = self._insert(node.left, city)
        else:
            node.right = self._insert(node.right, city)

        self._update_height(node)
        balance = self._bf(node)

        # Left Left
        if balance > 1 and city.distance < node.left.city.distance:
            return self._rotate_right(node)
        # Right Right
        if balance < -1 and city.distance >= node.right.city.distance:
            return self._rotate_left(node)
        # Left Right
        if balance > 1 and city.distance >= node.left.city.distance:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        # Right Left
        if balance < -1 and city.distance < node.right.city.distance:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def search(self, distance: float):
        node = self.root
        while node is not None:
            if distance == node.city.distance:
                return node.city
            node = node.left if distance < node.city.distance else node.right
        return None

    def delete(self, distance: float):
        before = self._size
        self.root = self._delete(self.root, distance)
        return self._size < before

    def _delete(self, node, distance):
        if node is None:
            return None
        if distance < node.city.distance:
            node.left = self._delete(node.left, distance)
        elif distance > node.city.distance:
            node.right = self._delete(node.right, distance)
        else:
            self._size -= 1
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            succ = node.right
            while succ.left is not None:
                succ = succ.left
            node.city = succ.city
            # remove successor from right subtree without double-decrementing size
            self._size += 1  # cancel the decrement that will happen inside recursive call
            node.right = self._delete(node.right, succ.city.distance)

        if node is None:
            return node

        self._update_height(node)
        balance = self._bf(node)

        if balance > 1 and self._bf(node.left) >= 0:
            return self._rotate_right(node)
        if balance > 1 and self._bf(node.left) < 0:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1 and self._bf(node.right) <= 0:
            return self._rotate_left(node)
        if balance < -1 and self._bf(node.right) > 0:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def height(self):
        return self._h(self.root) - 1 if self.root else -1


# ---------------------------------------------------------------------------
# 3. MIN-HEAP (array based, priority queue for "next nearest city")
# ---------------------------------------------------------------------------
class MinHeap:
    """Binary min-heap keyed on City.distance. Supports O(log n) insert/extract-min."""

    def __init__(self):
        self.data = []  # list[City]

    def __len__(self):
        return len(self.data)

    def _parent(self, i):
        return (i - 1) // 2

    def _left(self, i):
        return 2 * i + 1

    def _right(self, i):
        return 2 * i + 2

    def insert(self, city: City):
        self.data.append(city)
        self._sift_up(len(self.data) - 1)

    def _sift_up(self, i):
        while i > 0 and self.data[self._parent(i)].distance > self.data[i].distance:
            p = self._parent(i)
            self.data[i], self.data[p] = self.data[p], self.data[i]
            i = p

    def peek(self):
        return self.data[0] if self.data else None

    def extract_min(self):
        if not self.data:
            return None
        root = self.data[0]
        last = self.data.pop()
        if self.data:
            self.data[0] = last
            self._sift_down(0)
        return root

    def _sift_down(self, i):
        n = len(self.data)
        while True:
            l, r = self._left(i), self._right(i)
            smallest = i
            if l < n and self.data[l].distance < self.data[smallest].distance:
                smallest = l
            if r < n and self.data[r].distance < self.data[smallest].distance:
                smallest = r
            if smallest == i:
                break
            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]
            i = smallest

    def search(self, distance: float):
        # O(n) - heaps give no sub-linear search guarantee (documented limitation)
        for c in self.data:
            if c.distance == distance:
                return c
        return None

    @classmethod
    def build_heap(cls, cities):
        """O(n) build via bottom-up heapify, instead of n x O(log n) inserts."""
        h = cls()
        h.data = list(cities)
        n = len(h.data)
        for i in range(n // 2 - 1, -1, -1):
            h._sift_down(i)
        return h


# ---------------------------------------------------------------------------
# 4. HASH TABLE (separate chaining collision resolution)
# ---------------------------------------------------------------------------
class HashTable:
    """Hash table keyed on city name (string), separate chaining via linked lists (python lists)."""

    def __init__(self, capacity: int = 16, load_factor_threshold: float = 0.75):
        self.capacity = capacity
        self.size = 0
        self.load_factor_threshold = load_factor_threshold
        self.buckets = [[] for _ in range(capacity)]

    def __len__(self):
        return self.size

    def _hash(self, key: str) -> int:
        # Polynomial rolling hash (like Java's String.hashCode), then division method
        h = 0
        for ch in key:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h % self.capacity

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old_buckets:
            for key, value in bucket:
                self.insert(key, value)

    def insert(self, key: str, value: City):
        if (self.size + 1) / self.capacity > self.load_factor_threshold:
            self._resize()
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.size += 1

    def search(self, key: str):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return None

    def delete(self, key: str):
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False


# ---------------------------------------------------------------------------
# Helper: synthetic city generator
# ---------------------------------------------------------------------------
def generate_cities(n: int, seed: int = 42):
    rng = random.Random(seed)
    cities = []
    for i in range(n):
        lat = rng.uniform(-90, 90)
        lon = rng.uniform(-180, 180)
        pop = rng.randint(1_000, 5_000_000)
        distance = rng.uniform(0, 20000)  # km from reference origin, unique-ish
        cities.append(City(name=f"City_{i}_{rng.randint(0,999999)}", lat=lat, lon=lon,
                            population=pop, distance=distance))
    return cities


if __name__ == "__main__":
    # Quick sanity check
    cities = generate_cities(20)

    bst = BST()
    avl = AVLTree()
    ht = HashTable()
    for c in cities:
        bst.insert(c)
        avl.insert(c)
        ht.insert(c.name, c)

    heap = MinHeap.build_heap(cities)

    print("BST height (n=20):", bst.height())
    print("AVL height (n=20):", avl.height())
    print("Heap peek (should be min distance):", heap.peek())
    print("Hash lookup test:", ht.search(cities[5].name) == cities[5])

    # Test deletion
    target = cities[3].distance
    print("BST delete:", bst.delete(target))
    print("AVL delete:", avl.delete(target))
    print("BST search after delete (should be None):", bst.search(target))