"""
Lab Work 2: Prefix Averages - Quadratic versus Linear-Time Design
-----------------------------------------------------------------
Given a sequence S of n numbers, A[j] = (S[0] + ... + S[j]) / (j + 1).

Usage:
    python lab2_prefix_averages.py          # tests + full benchmark
    python lab2_prefix_averages.py --quick  # tests + small benchmark

Outputs (in ./lab2_output): timing_table.md, timing_data.csv, plot_*.png

DERIVATIONS (primitive operation = one addition of an element to a sum)
-----------------------------------------------------------------------
prefix_average1 (recomputes each prefix sum):
    for j in 0..n-1: inner loop adds j+1 elements
    additions = sum_{j=0}^{n-1} (j+1) = sum_{k=1}^{n} k = n(n+1)/2
    plus n divisions and n appends (O(n)).
    T1(n) = n(n+1)/2 + O(n) = n^2/2 + O(n)   ->  Theta(n^2)

prefix_average2 (running sum):
    each element is added exactly once: n additions, n divisions, n appends.
    T2(n) = c*n                              ->  Theta(n)

Auxiliary space: both use O(n) for the output list and O(1) extra.

Empty input: both functions return [] (documented design decision; there is
no prefix to average, and raising would make the empty case a special error).
"""
import csv
import math
import os
import platform
import random
import statistics
import sys
import time

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

SEED = 2024
TRIALS = 5
OUT_DIR = "lab2_output"


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------
def prefix_average1(S):
    """Quadratic: recompute each prefix sum from scratch."""
    n = len(S)
    A = [0.0] * n
    for j in range(n):
        total = 0
        for i in range(j + 1):          # j+1 additions
            total += S[i]
        A[j] = total / (j + 1)
    return A


def prefix_average2(S):
    """Linear: maintain a running sum."""
    n = len(S)
    A = [0.0] * n
    total = 0
    for j in range(n):
        total += S[j]                   # one addition per element
        A[j] = total / (j + 1)
    return A


def count_additions_1(n):
    """Exact addition count of prefix_average1 = n(n+1)/2."""
    return n * (n + 1) // 2


def count_additions_2(n):
    """Exact addition count of prefix_average2 = n."""
    return n


# --------------------------------------------------------------------------
# Validation tests
# --------------------------------------------------------------------------
def _close(a, b):
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9) for x, y in zip(a, b))


def run_tests():
    cases = {
        "empty": ([], []),
        "singleton": ([5], [5.0]),
        "positive": ([1, 2, 3, 4], [1.0, 1.5, 2.0, 2.5]),
        "negative": ([-2, -4, -6], [-2.0, -3.0, -4.0]),
        "mixed": ([10, -10, 5, -5, 0], [10.0, 0.0, 5 / 3, 0.0, 0.0]),
        "floats": ([0.5, 1.5, 2.5], [0.5, 1.0, 1.5]),
    }
    for name, (seq, expected) in cases.items():
        r1, r2 = prefix_average1(seq), prefix_average2(seq)
        assert _close(r1, expected), f"prefix_average1 failed on {name}"
        assert _close(r2, expected), f"prefix_average2 failed on {name}"
        assert _close(r1, r2), f"implementations disagree on {name}"
        print(f"  [PASS] {name:<10} -> {r2}")

    # Randomised equivalence check
    rng = random.Random(SEED)
    for _ in range(200):
        seq = [rng.randint(-1000, 1000) for _ in range(rng.randint(0, 50))]
        assert _close(prefix_average1(seq), prefix_average2(seq))
    print("  [PASS] 200 random sequences: both implementations agree")

    # Operation-count formulas
    assert count_additions_1(4) == 10 and count_additions_2(4) == 4
    print("  [PASS] operation-count formulas n(n+1)/2 and n")


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def median_time(func, S, trials=TRIALS):
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        func(S)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def benchmark(sizes_slow, sizes_fast):
    """Input is generated OUTSIDE the timed region (seeded, reproducible)."""
    rng = random.Random(SEED)
    data1, data2 = [], []
    for n in sorted(set(sizes_slow) | set(sizes_fast)):
        S = [rng.uniform(-100, 100) for _ in range(n)]
        if n in sizes_slow:
            data1.append((n, median_time(prefix_average1, S)))
        if n in sizes_fast:
            data2.append((n, median_time(prefix_average2, S)))
        print(f"  n={n:<8} done")
    return data1, data2


def precision_demo():
    """Compare float results of both methods on ill-conditioned input, with
    math.fsum as an (almost) exact reference."""
    rng = random.Random(SEED)
    n = 2000
    S = [rng.choice([1e16, -1e16, 1.0, 0.1]) for _ in range(n)]
    ref = [math.fsum(S[:j + 1]) / (j + 1) for j in range(n)]
    a1, a2 = prefix_average1(S), prefix_average2(S)
    e1 = max(abs(x - y) for x, y in zip(a1, ref))
    e2 = max(abs(x - y) for x, y in zip(a2, ref))
    print(f"  max abs error vs fsum  (method 1): {e1:.3e}")
    print(f"  max abs error vs fsum  (method 2): {e2:.3e}")
    print("  Note: both use naive float summation, so errors are similar; the\n"
          "  running sum accumulates rounding over ONE long chain, whereas\n"
          "  method 1 restarts each prefix (errors do not carry between j).")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def report(d1, d2):
    os.makedirs(OUT_DIR, exist_ok=True)
    t1, t2 = dict(d1), dict(d2)
    common = sorted(set(t1) & set(t2))

    with open(os.path.join(OUT_DIR, "timing_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "prefix_average1_s", "prefix_average2_s"])
        for n in sorted(set(t1) | set(t2)):
            w.writerow([n, t1.get(n, ""), t2.get(n, "")])

    lines = ["| n | prefix_average1 (s) | prefix_average2 (s) | speed-up | "
             "T1/n^2 | T2/n |", "|---|---|---|---|---|---|"]
    for n in common:
        lines.append(f"| {n} | {t1[n]:.3e} | {t2[n]:.3e} | "
                     f"{t1[n] / t2[n]:.1f}x | {t1[n] / n**2:.2e} | "
                     f"{t2[n] / n:.2e} |")
    table = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "timing_table.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)

    if not HAVE_MPL:
        print("matplotlib not installed - skipping plots.")
        return

    # Linear axes
    plt.figure(figsize=(7, 4.5))
    plt.plot(*zip(*d1), "o-", label="prefix_average1  O(n^2)")
    plt.plot(*zip(*d2), "s-", label="prefix_average2  O(n)")
    plt.xlabel("n"); plt.ylabel("median time (s)")
    plt.title("Prefix averages: execution time vs n")
    plt.legend(); plt.grid(True, alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_time_vs_n.png"), dpi=130)
    plt.close()

    # Log-log with reference slopes
    plt.figure(figsize=(7, 4.5))
    plt.loglog(*zip(*d1), "o-", label="prefix_average1")
    plt.loglog(*zip(*d2), "s-", label="prefix_average2")
    n0, tt1 = d1[0]
    ns = [n for n, _ in d1]
    plt.loglog(ns, [tt1 * (n / n0) ** 2 for n in ns], "k--", alpha=.5,
               label="reference slope 2 (n^2)")
    n0b, tt2 = d2[0]
    ns2 = [n for n, _ in d2]
    plt.loglog(ns2, [tt2 * (n / n0b) for n in ns2], "k:", alpha=.5,
               label="reference slope 1 (n)")
    plt.xlabel("n"); plt.ylabel("median time (s)")
    plt.title("Log-log: slope reveals the exponent")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_loglog.png"), dpi=130)
    plt.close()

    # Normalised
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([n for n, _ in d1], [t / n ** 2 for n, t in d1], "o-")
    ax[0].set_title("prefix_average1: T(n)/n^2 (flat => quadratic)")
    ax[1].plot([n for n, _ in d2], [t / n for n, t in d2], "s-")
    ax[1].set_title("prefix_average2: T(n)/n (flat => linear)")
    for a in ax:
        a.set_xscale("log"); a.grid(True, alpha=.3); a.set_xlabel("n")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot_normalised.png"), dpi=130)
    plt.close(fig)
    print(f"Plots saved in ./{OUT_DIR}/")


def main():
    quick = "--quick" in sys.argv
    print("=== Lab 2: Prefix Averages ===")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"Trials   : {TRIALS} (reported value = MEDIAN), seed = {SEED}\n")

    print("Validation tests:")
    run_tests()

    print("\nBenchmarking:")
    if quick:
        slow = [10, 50, 100, 200, 400]
        fast = slow + [1000, 5000]
    else:
        slow = [10, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
        fast = slow + [16000, 64000, 256000, 1000000]
    d1, d2 = benchmark(slow, fast)
    report(d1, d2)

    print("\nNumerical precision:")
    precision_demo()

    print("\nObservation: the linear method is already faster at tiny n "
          "(fixed overhead is\nsimilar, but method 1 does n(n+1)/2 "
          "additions vs n), and the speed-up grows\nroughly like (n+1)/2 "
          "as n increases.")


if __name__ == "__main__":
    main()
