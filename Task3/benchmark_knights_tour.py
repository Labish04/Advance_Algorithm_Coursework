"""
Benchmarks naive backtracking vs Warnsdorff-heuristic Knight's Tour across
increasing board sizes, measuring both wall-clock time and the number of
recursive calls made (a size-independent measure of search-space explored).
"""
import time
import csv
from backtracking_knights_tour import knights_tour_naive, knights_tour_warnsdorff, verify_tour

CALL_LIMIT = 2_000_000  # safety cap so naive search doesn't run forever on larger n


def run_all():
    rows = []
    for n in [5, 6, 7, 8]:
        # Warnsdorff
        start = time.perf_counter()
        path_w, stats_w = knights_tour_warnsdorff(n)
        t_w = time.perf_counter() - start
        valid_w = verify_tour(path_w, n)

        # Naive (capped) -- for n>=6 this can explode combinatorially
        start = time.perf_counter()
        path_n, stats_n = knights_tour_naive(n, call_limit=CALL_LIMIT)
        t_n = time.perf_counter() - start
        valid_n = verify_tour(path_n, n) if path_n else False
        hit_limit = stats_n.calls > CALL_LIMIT

        rows.append(dict(n=n,
                          warnsdorff_calls=stats_w.calls, warnsdorff_backtracks=stats_w.backtracks,
                          warnsdorff_time=t_w, warnsdorff_found=path_w is not None, warnsdorff_valid=valid_w,
                          naive_calls=stats_n.calls, naive_backtracks=stats_n.backtracks,
                          naive_time=t_n, naive_found=path_n is not None, naive_valid=valid_n,
                          naive_hit_limit=hit_limit))
        print(f"n={n}: Warnsdorff calls={stats_w.calls:>8} time={t_w:.4f}s | "
              f"Naive calls={stats_n.calls:>10} time={t_n:.4f}s hit_limit={hit_limit}")

    with open("knights_tour_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved knights_tour_results.csv")
    return rows


if __name__ == "__main__":
    run_all()