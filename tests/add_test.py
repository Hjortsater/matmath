import time
import numpy as np
import matplotlib.pyplot as plt; import matplotlib; matplotlib.use('TkAgg')

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hjortMatrix import Matrix as M

sizes = list(range(100, 2501, 300))
runs_per_size = 7

hjort_st_times = []
hjort_mt_times = []
numpy_times = []

for n in sizes:
    print(f"Testing addition for {n}x{n} matrices...")
    
    A_data = np.random.rand(n, n)
    B_data = np.random.rand(n, n)

    hjort_st_elapsed = 0.0
    for _ in range(runs_per_size):
        A = M(*A_data.tolist(), multithreaded=False)
        B = M(*B_data.tolist(), multithreaded=False)
        start = time.time()
        C = A + B
        hjort_st_elapsed += time.time() - start
    hjort_st_times.append(hjort_st_elapsed / runs_per_size)

    hjort_mt_elapsed = 0.0
    for _ in range(runs_per_size):
        A = M(*A_data.tolist(), multithreaded=True)
        B = M(*B_data.tolist(), multithreaded=True)
        start = time.time()
        C = A + B
        hjort_mt_elapsed += time.time() - start
    hjort_mt_times.append(hjort_mt_elapsed / runs_per_size)
    
    numpy_elapsed = 0.0
    for _ in range(runs_per_size):
        start = time.time()
        C_np = A_data + B_data
        numpy_elapsed += time.time() - start
    numpy_times.append(numpy_elapsed / runs_per_size)

plt.figure(figsize=(12, 7))
plt.plot(sizes, hjort_st_times, 'o-', label='HjortMatrix ST')
plt.plot(sizes, hjort_mt_times, 'o-', label='HjortMatrix MT')
plt.plot(sizes, numpy_times, 's-', label='NumPy')
plt.xlabel("Matrix size (n x n)")
plt.ylabel("Average time (seconds)")
plt.title(f"Matrix Addition Speed Comparison ({runs_per_size} runs averaged)")
plt.legend()
plt.grid(True)
plt.show()