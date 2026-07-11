"""Correctness tests for Knight's Tour: verifies tours are valid (Hamiltonian
knight-move paths) across multiple board sizes and starting squares."""
from backtracking_knights_tour import knights_tour_warnsdorff, knights_tour_naive, verify_tour


def test_warnsdorff_finds_valid_tour_various_sizes():
    for n in [5, 6, 7, 8, 10, 12]:
        path, stats = knights_tour_warnsdorff(n)
        assert path is not None, f"No tour found for n={n}"
        assert verify_tour(path, n), f"Invalid tour returned for n={n}"
    print("[PASS] Warnsdorff finds a valid, verified tour for n in {5,6,7,8,10,12}")


def test_warnsdorff_various_start_squares():
    n = 8
    for start in [(0, 0), (3, 3), (7, 7), (0, 7)]:
        path, stats = knights_tour_warnsdorff(n, start=start)
        assert path is not None, f"No tour found starting at {start}"
        assert path[0] == start
        assert verify_tour(path, n), f"Invalid tour for start={start}"
    print("[PASS] Warnsdorff finds valid tours from multiple starting squares (n=8)")


def test_naive_matches_warnsdorff_on_small_board():
    """On a small board both should find *a* valid tour (not necessarily the
    same one -- Knight's Tour is not unique) -- we only check validity."""
    n = 5
    path_naive, _ = knights_tour_naive(n)
    path_w, _ = knights_tour_warnsdorff(n)
    assert verify_tour(path_naive, n)
    assert verify_tour(path_w, n)
    print("[PASS] Both naive and Warnsdorff produce valid (independently verified) tours for n=5")


def test_verify_tour_rejects_invalid_paths():
    # wrong length
    assert not verify_tour([(0, 0), (1, 2)], 5)
    # illegal move (not an L-shape)
    assert not verify_tour([(0, 0), (0, 1)] + [(i, i) for i in range(23)], 5)
    # duplicate square
    n = 5
    bad_path = [(0, 0)] * (n * n)
    assert not verify_tour(bad_path, n)
    print("[PASS] verify_tour correctly rejects malformed/invalid tours")


if __name__ == "__main__":
    test_warnsdorff_finds_valid_tour_various_sizes()
    test_warnsdorff_various_start_squares()
    test_naive_matches_warnsdorff_on_small_board()
    test_verify_tour_rejects_invalid_paths()
    print("\nAll Knight's Tour correctness tests passed.")