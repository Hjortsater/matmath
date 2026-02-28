import time
import numpy as np
import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('TkAgg')

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hjortMatrix import Matrix as M


# -----------------------------
# Settings
# -----------------------------
sizes = list(range(100, 3001, 300))
runs_per_size = 7  # reduced
np.random.seed(42)

plt.style.use("dark_background")


# -----------------------------
# Outlier Removal (IQR method)
# -----------------------------
def remove_outliers(data):
    """
    data: 1D array
    returns filtered array
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return data[(data >= lower) & (data <= upper)]


# -----------------------------
# Benchmark Function
# -----------------------------
def benchmark(operation_name):
    hjort_raw = []
    numpy_raw = []

    for n in sizes:
        print(f"{operation_name.upper()}: {n}x{n}")

        A_data = np.random.rand(n, n)
        B_data = np.random.rand(n, n)

        hjort_runs = []
        numpy_runs = []

        for _ in range(runs_per_size):

            # Hjort
            A = M(*A_data.tolist())
            B = M(*B_data.tolist())
            start = time.time()

            if operation_name == "add":
                _ = A + B
            elif operation_name == "sub":
                _ = A - B
            elif operation_name == "mul":
                _ = A * B

            hjort_runs.append(time.time() - start)

            # NumPy
            start = time.time()
            if operation_name == "add":
                _ = A_data + B_data
            elif operation_name == "sub":
                _ = A_data - B_data
            elif operation_name == "mul":
                _ = np.dot(A_data, B_data)

            numpy_runs.append(time.time() - start)

        # Remove outliers per size
        hjort_raw.append(remove_outliers(np.array(hjort_runs)))
        numpy_raw.append(remove_outliers(np.array(numpy_runs)))

    return hjort_raw, numpy_raw


# -----------------------------
# Plotting helper
# -----------------------------
def plot_with_fit(ax, sizes, raw_data, label, color, degree):
    means = []
    stds = []

    for arr in raw_data:
        means.append(arr.mean())
        stds.append(arr.std())

    means = np.array(means)
    stds = np.array(stds)

    # Polynomial fit
    coeffs = np.polyfit(sizes, means, degree)
    poly = np.poly1d(coeffs)

    smooth_x = np.linspace(min(sizes), max(sizes), 400)
    smooth_y = poly(smooth_x)

    # Raw points
    for i, size in enumerate(sizes):
        ax.scatter(
            [size] * len(raw_data[i]),
            raw_data[i],
            color=color,
            alpha=0.25,
            s=18
        )

    # Mean markers
    ax.plot(sizes, means, 'o', color=color, label=f"{label} mean")

    # Smooth curve
    ax.plot(smooth_x, smooth_y, '-', color=color, linewidth=2.5)

    # Spread cone (±1σ)
    ax.fill_between(
        sizes,
        means - stds,
        means + stds,
        color=color,
        alpha=0.18
    )


# -----------------------------
# Run benchmarks
# -----------------------------
add_hjort, add_numpy = benchmark("add")
sub_hjort, sub_numpy = benchmark("sub")
mul_hjort, mul_numpy = benchmark("mul")


# -----------------------------
# Plotting
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(20, 7))

# Addition
plot_with_fit(axs[0], sizes, add_hjort, "HjortMatrix", "#00BFFF", degree=2)
plot_with_fit(axs[0], sizes, add_numpy, "NumPy", "#FF4C4C", degree=2)
axs[0].set_title("Addition (≈ O(n²))", fontsize=14)
axs[0].set_xlabel("Matrix Size (n x n)")
axs[0].set_ylabel("Time (seconds)")
axs[0].legend()

# Subtraction
plot_with_fit(axs[1], sizes, sub_hjort, "HjortMatrix", "#00BFFF", degree=2)
plot_with_fit(axs[1], sizes, sub_numpy, "NumPy", "#FF4C4C", degree=2)
axs[1].set_title("Subtraction (≈ O(n²))", fontsize=14)
axs[1].set_xlabel("Matrix Size (n x n)")
axs[1].legend()

# Multiplication
plot_with_fit(axs[2], sizes, mul_hjort, "HjortMatrix", "#00BFFF", degree=3)
plot_with_fit(axs[2], sizes, mul_numpy, "NumPy", "#FF4C4C", degree=3)
axs[2].set_title("Multiplication (≈ O(n³))", fontsize=14)
axs[2].set_xlabel("Matrix Size (n x n)")
axs[2].legend()

plt.suptitle(
    "Complete Matrix Benchmark (7 Runs, IQR Outlier Removal)\n"
    "Dots = Filtered Raw Data | Line = Polynomial Fit | Shaded = ±1σ",
    fontsize=16
)

plt.tight_layout()
plt.show()