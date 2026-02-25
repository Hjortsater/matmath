from hjortmath import Matrix
import numpy as np
import statistics
import time
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    def to_matrix(arr: np.ndarray, use_C: bool = True, multithreaded: bool = True) -> Matrix:
        return Matrix(*[tuple(map(float, row)) for row in arr], use_C=use_C, multithreaded=multithreaded)

    def max_abs_diff_matrix(mat: Matrix, np_arr: np.ndarray) -> float:
        return float(np.max(np.abs(np.array(mat._to_tuple_form()) - np_arr)))

    def run_benchmark(size: int, iterations: int = 3) -> None:
        add_err, mul_err, inv_err = [], [], []
        det_abs_err, det_rel_err = [], []

        add_time_py_mt, add_time_py_st, add_time_np = [], [], []
        mul_time_py_mt, mul_time_py_st, mul_time_np = [], [], []
        inv_time_py_mt, inv_time_py_st, inv_time_np = [], [], []
        det_time_py_mt, det_time_py_st, det_time_np = [], [], []

        for _ in range(iterations):
            a_np = np.random.rand(size, size)
            b_np = np.random.rand(size, size)

            # Multithreaded tests
            A_mt = to_matrix(a_np, use_C=True, multithreaded=True)
            B_mt = to_matrix(b_np, use_C=True, multithreaded=True)

            start = time.perf_counter()
            C_mt = A_mt + B_mt
            add_time_py_mt.append(time.perf_counter() - start)

            start = time.perf_counter()
            M_mt = A_mt * B_mt
            mul_time_py_mt.append(time.perf_counter() - start)

            start = time.perf_counter()
            inv_A_mt = A_mt.inverse
            inv_time_py_mt.append(time.perf_counter() - start)

            start = time.perf_counter()
            det_A_mt = float(A_mt.determinant)
            det_time_py_mt.append(time.perf_counter() - start)

            # Single-threaded tests
            A_st = to_matrix(a_np, use_C=True, multithreaded=False)
            B_st = to_matrix(b_np, use_C=True, multithreaded=False)

            start = time.perf_counter()
            C_st = A_st + B_st
            add_time_py_st.append(time.perf_counter() - start)

            start = time.perf_counter()
            M_st = A_st * B_st
            mul_time_py_st.append(time.perf_counter() - start)

            start = time.perf_counter()
            inv_A_st = A_st.inverse
            inv_time_py_st.append(time.perf_counter() - start)

            start = time.perf_counter()
            det_A_st = float(A_st.determinant)
            det_time_py_st.append(time.perf_counter() - start)

            # NumPy baseline
            start = time.perf_counter()
            C_np = a_np + b_np
            add_time_np.append(time.perf_counter() - start)

            start = time.perf_counter()
            M_np = a_np @ b_np
            mul_time_np.append(time.perf_counter() - start)

            start = time.perf_counter()
            inv_np = np.linalg.inv(a_np)
            inv_time_np.append(time.perf_counter() - start)

            start = time.perf_counter()
            det_np = float(np.linalg.det(a_np))
            det_time_np.append(time.perf_counter() - start)

            # Calculate errors (using multithreaded results as they should be identical)
            add_err.append(max_abs_diff_matrix(C_mt, C_np))
            mul_err.append(max_abs_diff_matrix(M_mt, M_np))
            inv_err.append(max_abs_diff_matrix(inv_A_mt, inv_np))
            
            abs_err = abs(det_A_mt - det_np)
            rel_err = abs_err / max(1.0, abs(det_np))
            det_abs_err.append(abs_err)
            det_rel_err.append(rel_err)

        print(f"\n{'='*80}")
        print(f"Matrix Size: {size}x{size} (iterations: {iterations})")
        print(f"{'='*80}")
        
        print("\nAccuracy:")
        print(f"{'Operation':<15} | {'Max Abs Error':<15} | {'Mean Abs Error'}")
        print("-" * 60)
        print(f"{'Addition':<15} | {max(add_err):<15.6e} | {statistics.mean(add_err):.6e}")
        print(f"{'Multiplication':<15} | {max(mul_err):<15.6e} | {statistics.mean(mul_err):.6e}")
        print(f"{'Inversion':<15} | {max(inv_err):<15.6e} | {statistics.mean(inv_err):.6e}")
        print(f"{'Det (abs)':<15} | {max(det_abs_err):<15.6e} | {statistics.mean(det_abs_err):.6e}")
        print(f"{'Det (rel)':<15} | {max(det_rel_err):<15.6e} | {statistics.mean(det_rel_err):.6e}")

        print("\nSpeed (mean seconds):")
        print(f"{'Operation':<15} | {'MT':<12} | {'ST':<12} | {'NumPy':<12} | {'MT/ST':<10} | {'MT/NP'}")
        print("-" * 80)

        def speedup(fast, slow):
            return f"{(slow/fast):.2f}x" if fast > 0 else "N/A"

        add_py_mt = statistics.mean(add_time_py_mt)
        add_py_st = statistics.mean(add_time_py_st)
        add_np = statistics.mean(add_time_np)
        
        mul_py_mt = statistics.mean(mul_time_py_mt)
        mul_py_st = statistics.mean(mul_time_py_st)
        mul_np = statistics.mean(mul_time_np)
        
        inv_py_mt = statistics.mean(inv_time_py_mt)
        inv_py_st = statistics.mean(inv_time_py_st)
        inv_np = statistics.mean(inv_time_np)
        
        det_py_mt = statistics.mean(det_time_py_mt)
        det_py_st = statistics.mean(det_time_py_st)
        det_np = statistics.mean(det_time_np)

        print(f"{'Addition':<15} | {add_py_mt:<12.6f} | {add_py_st:<12.6f} | {add_np:<12.6f} | {speedup(add_py_st, add_py_mt):<10} | {speedup(add_np, add_py_mt)}")
        print(f"{'Multiplication':<15} | {mul_py_mt:<12.6f} | {mul_py_st:<12.6f} | {mul_np:<12.6f} | {speedup(mul_py_st, mul_py_mt):<10} | {speedup(mul_np, mul_py_mt)}")
        print(f"{'Inversion':<15} | {inv_py_mt:<12.6f} | {inv_py_st:<12.6f} | {inv_np:<12.6f} | {speedup(inv_py_st, inv_py_mt):<10} | {speedup(inv_np, inv_py_mt)}")
        print(f"{'Determinant':<15} | {det_py_mt:<12.6f} | {det_py_st:<12.6f} | {det_np:<12.6f} | {speedup(det_py_st, det_py_mt):<10} | {speedup(det_np, det_py_mt)}")

        # Calculate parallel efficiency
        print("\nParallel Speedup (MT vs ST):")
        print(f"{'Operation':<15} | {'Speedup':<10} | {'Efficiency'}")
        print("-" * 40)
        
        add_speedup = add_py_st / add_py_mt
        mul_speedup = mul_py_st / mul_py_mt
        inv_speedup = inv_py_st / inv_py_mt
        det_speedup = det_py_st / det_py_mt
        
        # Rough estimate of available parallelism (assuming 4 cores for efficiency calculation)
        cores = 4
        print(f"{'Addition':<15} | {add_speedup:<10.2f}x | {add_speedup/cores:<10.2%}")
        print(f"{'Multiplication':<15} | {mul_speedup:<10.2f}x | {mul_speedup/cores:<10.2%}")
        print(f"{'Inversion':<15} | {inv_speedup:<10.2f}x | {inv_speedup/cores:<10.2%}")
        print(f"{'Determinant':<15} | {det_speedup:<10.2f}x | {det_speedup/cores:<10.2%}")

    # Run benchmarks with different sizes
    print("MULTITHREADED VS SINGLE-THREADED PERFORMANCE COMPARISON")
    print("="*80)
    
    run_benchmark(10, 5)
    run_benchmark(25, 5)
    run_benchmark(50, 3)
    run_benchmark(100, 2)
    run_benchmark(200, 1)
    
    # Additional test for very large matrix to highlight parallel advantage
    print("\n" + "="*80)
    print("LARGE MATRIX TEST (showing parallel advantage)")
    print("="*80)
    run_benchmark(500, 1)