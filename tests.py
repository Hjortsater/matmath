from hjortmath import Matrix
import numpy as np
import statistics
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    def to_matrix(arr: np.ndarray, use_C: bool = True) -> Matrix:
        return Matrix(*[tuple(map(float, row)) for row in arr], use_C=use_C)

    def max_abs_diff_matrix(mat: Matrix, np_arr: np.ndarray) -> float:
        return float(np.max(np.abs(np.array(mat._to_tuple_form()) - np_arr)))

    def run_accuracy_test(size: int, iterations: int = 5) -> None:
        add_err, mul_err, inv_err = [], [], []
        det_abs_err, det_rel_err = [], []

        for _ in range(iterations):
            a_np = np.random.rand(size, size)
            b_np = np.random.rand(size, size)

            A = to_matrix(a_np, use_C=True)
            B = to_matrix(b_np, use_C=True)

            C = A + B
            add_err.append(max_abs_diff_matrix(C, a_np + b_np))

            M = A * B
            mul_err.append(max_abs_diff_matrix(M, a_np @ b_np))

            inv_np = np.linalg.inv(a_np)
            inv_A = A.inverse
            inv_err.append(max_abs_diff_matrix(inv_A, inv_np))

            det_np = float(np.linalg.det(a_np))
            det_A = float(A.determinant)

            abs_err = abs(det_A - det_np)
            rel_err = abs_err / max(1.0, abs(det_np))

            det_abs_err.append(abs_err)
            det_rel_err.append(rel_err)

        print(f"\nMatrix Size: {size}x{size}")
        print(f"{'Operation':<15} | {'Max Abs Error':<15} | {'Mean Abs Error'}")
        print("-" * 60)
        print(f"{'Addition':<15} | {max(add_err):<15.6e} | {statistics.mean(add_err):.6e}")
        print(f"{'Multiplication':<15} | {max(mul_err):<15.6e} | {statistics.mean(mul_err):.6e}")
        print(f"{'Inversion':<15} | {max(inv_err):<15.6e} | {statistics.mean(inv_err):.6e}")
        print(f"{'Det (abs)':<15} | {max(det_abs_err):<15.6e} | {statistics.mean(det_abs_err):.6e}")
        print(f"{'Det (rel)':<15} | {max(det_rel_err):<15.6e} | {statistics.mean(det_rel_err):.6e}")
        print(f"Actual determinant, Numpy: {det_np}, Custom: {det_A}. \n")

    run_accuracy_test(5, 10)
    run_accuracy_test(10, 10)
    run_accuracy_test(25, 5)
    run_accuracy_test(50, 3)
    run_accuracy_test(100, 2)