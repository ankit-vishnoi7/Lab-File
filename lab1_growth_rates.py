"""
Lab Work 1: Benchmarking Algorithmic Growth Rates
-------------------------------------------------
Empirically investigates constant, logarithmic, linear, n log n, quadratic
and exponential growth.

Usage:
    python lab1_growth_rates.py          # full run
    python lab1_growth_rates.py --quick  # small/fast run for testing

Outputs (in ./lab1_output):
    timing_data.csv, comparison_table.md, plot_*.png

Methodology:
  * Input generation/validation is OUTSIDE the timed region.
  * Each point is timed `TRIALS` times; the reported value is the MEDIAN
    (robust against OS scheduling noise / GC pauses).
  * Random inputs use a fixed seed -> reproducible.
  * Worst-case inputs are used where relevant (e.g. linear search for an
    absent key), documented in INPUT_NOTES.
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
except ImportError:  # plots are skipped, data is still produced
    HAVE_MPL = False

SEED = 12345
TRIALS = 7
OUT_DIR = "lab1_output"


# --------------------------------------------------------------------------
# Algorithms under test
# --------------------------------------------------------------------------
def list_index(data, idx):
    """O(1): random access by index."""
    return data[idx]


def binary_search(data, target):
    """O(log n): data must be sorted."""
    lo, hi = 0, len(data) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if data[mid] == target:
            return mid
        if data[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def linear_search(data, target):
    """O(n): scans every element in the worst case."""
    for i, v in enumerate(data):
        if v == target:
            return i
    return -1


def merge_sort(data):
    """O(n log n): top-down merge sort (returns a new list)."""
    if len(data) <= 1:
        return data[:]
    mid = len(data) // 2
    left, right = merge_sort(data[:mid]), merge_sort(data[mid:])
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def count_pairs(data, target):
    """O(n^2): count pairs (i<j) with data[i]+data[j]==target (double loop)."""
    n, count = len(data), 0
    for i in range(n):
        for j in range(i + 1, n):
            if data[i] + data[j] == target:
                count += 1
    return count


def enumerate_subsets(data):
    """O(2^n * n): visit every subset via bitmasks (small n only)."""
    n, total = len(data), 0
    for mask in range(1 << n):
        s = 0
        for k in range(n):
            if mask >> k & 1:
                s += data[k]
        total += s
    return total


# --------------------------------------------------------------------------
# Timing harness
# --------------------------------------------------------------------------
def time_median(func, args, trials=TRIALS, inner=1):
    """Median seconds per call. `inner` repeats the call to beat timer
    resolution for very fast operations; the result is divided by `inner`."""
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(inner):
            func(*args)
        samples.append((time.perf_counter() - t0) / inner)
    return statistics.median(samples)


def benchmark(name, func, make_args, sizes, inner_fn=lambda n: 1):
    """make_args(n) builds inputs (untimed). Returns list of (n, seconds)."""
    results = []
    for n in sizes:
        args = make_args(n)            # untimed input generation
        t = time_median(func, args, inner=inner_fn(n))
        results.append((n, t))
        print(f"  {name:<18} n={n:<9} {t:.3e} s")
    return results


# --------------------------------------------------------------------------
# Input builders (all seeded -> reproducible)
# --------------------------------------------------------------------------
def _rng():
    return random.Random(SEED)


def args_index(n):
    return list(range(n)), n // 2


def args_binary(n):
    return list(range(n)), -1                      # absent key -> worst case


def args_linear(n):
    return list(range(n)), -1                      # absent key -> worst case


def args_sort(n):
    r = _rng()
    return ([r.randint(0, 10 ** 9) for _ in range(n)],)


def args_pairs(n):
    r = _rng()
    return [r.randint(0, 1000) for _ in range(n)], -1   # no match: full scan


def args_subsets(n):
    return ([1] * n,)


# --------------------------------------------------------------------------
# Experiment definition: (name, func, args builder, sizes_full, sizes_quick,
#                         theoretical bound, normaliser, inner-repeat fn)
# --------------------------------------------------------------------------
def log2(n):
    return math.log2(n) if n > 1 else 1.0


EXPERIMENTS = [
    ("list indexing", list_index, args_index,
     [10 ** k for k in range(2, 7)], [100, 1000, 10000],
     "O(1)", lambda n: 1.0, lambda n: 20000),
    ("binary search", binary_search, args_binary,
     [2 ** k for k in range(10, 23, 2)], [1024, 4096, 16384],
     "O(log n)", log2, lambda n: 5000),
    ("linear search", linear_search, args_linear,
     [2 ** k for k in range(10, 21, 2)], [1024, 4096, 16384],
     "O(n)", lambda n: n, lambda n: 1),
    ("merge sort", merge_sort, args_sort,
     [2 ** k for k in range(8, 17)], [256, 1024, 4096],
     "O(n log n)", lambda n: n * log2(n), lambda n: 1),
    ("pair counter", count_pairs, args_pairs,
     [100, 200, 400, 800, 1600, 3200], [100, 200, 400],
     "O(n^2)", lambda n: n * n, lambda n: 1),
    ("subset enumeration", enumerate_subsets, args_subsets,
     list(range(4, 19, 2)), [4, 8, 12],
     "O(2^n)", lambda n: 2 ** n, lambda n: 1),
]

INPUT_NOTES = {
    "list indexing": "list(range(n)), middle index",
    "binary search": "sorted list(range(n)), absent key (worst case)",
    "linear search": "list(range(n)), absent key (worst case)",
    "merge sort": f"random ints in [0,1e9], seed={SEED}",
    "pair counter": f"random ints in [0,1000], seed={SEED}, no match (full scan)",
    "subset enumeration": "[1]*n, all 2^n subsets visited",
}

TREND = {
    "list indexing": "Flat; time independent of n",
    "binary search": "Grows very slowly (~ +const per doubling of n)",
    "linear search": "Doubles when n doubles",
    "merge sort": "Slightly super-linear; ~2.1x per doubling of n",
    "pair counter": "Quadruples when n doubles",
    "subset enumeration": "Roughly x4 per +2 in n (x2 per +1)",
}

LIMITS = {
    "list indexing": "Time ~ timer/loop overhead; cache effects at large n",
    "binary search": "Nanosecond scale; dominated by interpreter overhead",
    "linear search": "Early-exit not exercised; cache/memory effects",
    "merge sort": "Slicing/allocation cost; recursion overhead; GC noise",
    "pair counter": "n limited to a few thousand; interpreter overhead",
    "subset enumeration": "n <= ~18 only; extra factor n from inner loop",
}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def save_csv(all_results):
    path = os.path.join(OUT_DIR, "timing_data.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "n", "median_seconds"])
        for name, res in all_results.items():
            for n, t in res:
                w.writerow([name, n, f"{t:.9e}"])
    return path


def save_table(all_results):
    path = os.path.join(OUT_DIR, "comparison_table.md")
    bounds = {e[0]: e[5] for e in EXPERIMENTS}
    with open(path, "w") as f:
        f.write("| Algorithm | Theoretical bound | Measured trend | "
                "Experimental limitations |\n|---|---|---|---|\n")
        for name in all_results:
            f.write(f"| {name} | {bounds[name]} | {TREND[name]} | "
                    f"{LIMITS[name]} |\n")
        f.write("\n**Input characteristics**\n\n")
        for name in all_results:
            f.write(f"- {name}: {INPUT_NOTES[name]}\n")
    return path


def fit_exponent(res):
    """Slope of log T vs log n (empirical power-law exponent)."""
    xs = [math.log(n) for n, _ in res]
    ys = [math.log(t) for _, t in res]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0


def make_plots(all_results):
    if not HAVE_MPL:
        print("matplotlib not installed - skipping plots.")
        return
    norm = {e[0]: e[6] for e in EXPERIMENTS}

    # 1. One raw plot per algorithm
    for name, res in all_results.items():
        ns, ts = zip(*res)
        plt.figure(figsize=(6, 4))
        plt.plot(ns, ts, "o-")
        plt.xlabel("n"); plt.ylabel("median time (s)")
        plt.title(f"{name}  [{[e[5] for e in EXPERIMENTS if e[0]==name][0]}]")
        plt.grid(True, alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"plot_{name.replace(' ', '_')}.png"),
                    dpi=130)
        plt.close()

    # 2. Combined log-log plot
    plt.figure(figsize=(7, 5))
    for name, res in all_results.items():
        ns, ts = zip(*res)
        plt.loglog(ns, ts, "o-", label=name)
    plt.xlabel("n"); plt.ylabel("median time (s)")
    plt.title("All algorithms (log-log)")
    plt.legend(fontsize=8); plt.grid(True, which="both", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_all_loglog.png"), dpi=130)
    plt.close()

    # 3. Normalised plots: T(n)/f(n) should flatten if f matches the growth
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (name, res) in zip(axes.flat, all_results.items()):
        ns, ts = zip(*res)
        f = norm[name]
        ax.plot(ns, [t / f(n) for n, t in res], "o-")
        ax.set_title(f"{name}: T(n)/f(n)", fontsize=9)
        ax.set_xscale("log"); ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot_normalised.png"), dpi=130)
    plt.close(fig)

    # 4. The three standard normalisations on the polynomial-ish algorithms
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (label, f) in zip(axes, [("T(n)/n", lambda n: n),
                                     ("T(n)/(n log2 n)", lambda n: n * log2(n)),
                                     ("T(n)/n^2", lambda n: n * n)]):
        for name in ("linear search", "merge sort", "pair counter"):
            res = all_results[name]
            ax.plot([n for n, _ in res], [t / f(n) for n, t in res],
                    "o-", label=name)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(label); ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "plot_standard_normalisations.png"),
                dpi=130)
    plt.close(fig)


def main():
    quick = "--quick" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== Lab 1: Benchmarking Algorithmic Growth Rates ===")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"Trials   : {TRIALS} (reported value = MEDIAN), seed = {SEED}")
    print(f"Mode     : {'quick' if quick else 'full'}\n")

    all_results = {}
    for (name, func, make_args, full, small, _bound, _f, inner) in EXPERIMENTS:
        all_results[name] = benchmark(name, func, make_args,
                                      small if quick else full, inner)

    # Sanity checks (outside timing)
    data = list(range(1000))
    assert binary_search(data, 700) == 700 and linear_search(data, 700) == 700
    r = _rng(); arr = [r.randint(0, 99) for _ in range(500)]
    assert merge_sort(arr) == sorted(arr)
    assert count_pairs([1, 2, 3, 4], 5) == 2
    assert enumerate_subsets([1, 1, 1]) == 12      # 2^3 subsets, sum = 3*4

    print("\nEmpirical power-law exponent (slope of log T vs log n):")
    for name, res in all_results.items():
        if name != "subset enumeration":
            print(f"  {name:<18} {fit_exponent(res):+.2f}")

    print("\nSaved:", save_csv(all_results))
    print("Saved:", save_table(all_results))
    make_plots(all_results)
    print(f"Plots (if matplotlib available) in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
