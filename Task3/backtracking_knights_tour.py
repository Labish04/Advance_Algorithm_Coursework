"""
Task 3 - Backtracking: Knight's Tour
======================================
Find a sequence of knight moves that visits every square of an n x n board
exactly once (open tour; does not need to return to the start).

Two versions are implemented:
    1. Naive backtracking       - tries moves in a fixed fixed order, prunes
                                   only on "square already visited" / off-board.
    2. Warnsdorff's heuristic   - at each step, greedily move to the
                                   accessible square with the FEWEST onward
                                   moves (most-constrained-square-first).
                                   This is a heuristic move-ORDERING that
                                   dramatically increases the chance of the
                                   very first backtracking attempt succeeding
                                   (it still backtracks if it reaches a dead
                                   end, but in practice almost never needs to).

Both implementations count the number of recursive calls made, which is used
as a size-independent proxy for "amount of search space explored" -- the
key metric for evaluating how much a pruning strategy reduces the
exponential worst case in practice.
"""
import sys

sys.setrecursionlimit(20000)

MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]


def on_board(x, y, n):
    return 0 <= x < n and 0 <= y < n


class SearchStats:
    def __init__(self):
        self.calls = 0
        self.backtracks = 0


def knights_tour_naive(n, start=(0, 0), call_limit=None):
    """Naive backtracking, fixed move order. Returns (path_or_None, stats).
    call_limit: optional safety cap on recursive calls (naive search blows up
    exponentially for n >= 6, so this prevents runaway execution during
    benchmarking).
    """
    board = [[-1] * n for _ in range(n)]
    path = [start]
    board[start[0]][start[1]] = 0
    stats = SearchStats()

    def backtrack(x, y, move_count):
        stats.calls += 1
        if call_limit and stats.calls > call_limit:
            return "LIMIT"
        if move_count == n * n:
            return True
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if on_board(nx, ny, n) and board[nx][ny] == -1:
                board[nx][ny] = move_count
                path.append((nx, ny))
                result = backtrack(nx, ny, move_count + 1)
                if result is True:
                    return True
                if result == "LIMIT":
                    return "LIMIT"
                # undo (backtrack)
                board[nx][ny] = -1
                path.pop()
                stats.backtracks += 1
        return False

    result = backtrack(start[0], start[1], 1)
    if result is True:
        return list(path), stats
    return None, stats


def count_onward_moves(board, x, y, n):
    count = 0
    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if on_board(nx, ny, n) and board[nx][ny] == -1:
            count += 1
    return count


def knights_tour_warnsdorff(n, start=(0, 0), call_limit=None):
    """Backtracking with Warnsdorff's heuristic move ordering: at each step,
    try the onward square with the FEWEST further onward moves first (the
    most 'constrained' square, most likely to become unreachable later if
    not visited now). Still backtracks on dead ends (rare in practice for
    n >= 5), so correctness/completeness is unaffected -- only the search
    ORDER changes, which is what collapses the effective search space.
    """
    board = [[-1] * n for _ in range(n)]
    path = [start]
    board[start[0]][start[1]] = 0
    stats = SearchStats()

    def backtrack(x, y, move_count):
        stats.calls += 1
        if call_limit and stats.calls > call_limit:
            return "LIMIT"
        if move_count == n * n:
            return True

        candidates = []
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if on_board(nx, ny, n) and board[nx][ny] == -1:
                degree = count_onward_moves(board, nx, ny, n)
                candidates.append((degree, nx, ny))
        candidates.sort(key=lambda c: c[0])  # fewest onward moves first

        for degree, nx, ny in candidates:
            board[nx][ny] = move_count
            path.append((nx, ny))
            result = backtrack(nx, ny, move_count + 1)
            if result is True:
                return True
            if result == "LIMIT":
                return "LIMIT"
            board[nx][ny] = -1
            path.pop()
            stats.backtracks += 1
        return False

    result = backtrack(start[0], start[1], 1)
    if result is True:
        return list(path), stats
    return None, stats


def verify_tour(path, n):
    """Validates a claimed tour: correct length, all squares distinct and on
    board, and every consecutive pair is a legal knight move."""
    if path is None:
        return False
    if len(path) != n * n:
        return False
    if len(set(path)) != n * n:
        return False
    for (x, y) in path:
        if not on_board(x, y, n):
            return False
    for i in range(len(path) - 1):
        dx = abs(path[i][0] - path[i + 1][0])
        dy = abs(path[i][1] - path[i + 1][1])
        if sorted((dx, dy)) != [1, 2]:
            return False
    return True


if __name__ == "__main__":
    for n in [5, 6, 8]:
        path_w, stats_w = knights_tour_warnsdorff(n)
        valid = verify_tour(path_w, n)
        print(f"n={n}: Warnsdorff found tour={path_w is not None}, valid={valid}, "
              f"calls={stats_w.calls}, backtracks={stats_w.backtracks}")