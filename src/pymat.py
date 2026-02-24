import cmat
from typing import Self, Any, Union, List, Tuple
from customdecorators import *

class Matrix():
    """
    CUSTOM MATRIX CLASS IN PYTHON WITH A C BACKEND FOR EXPENSIVE COMPUTATIONS.

    WRITTEN FOR FUN, NOT FOR SPEED!
    """
    
    def __init__(self, *rows: Union[float, int, Tuple[Union[float, int], ...]], **kwargs: Any) -> None:
        """CONSTRUCTOR FOR MATRIX"""
        self.entries: List[float] = []
        self.m: int = 0  # Matrix height
        self.n: int = 0  # Matrix width

        self.use_C: bool = kwargs.get('use_C', True)
        self.force_C: bool = kwargs.get('force_C', False)
        self.use_color: bool = kwargs.get('use_color', True)
        self.sig_digits: int = kwargs.get('sig_digits', 4)
        self.disable_perf_hints: bool = kwargs.get('disable_warnings', False)

        if not rows:
            raise ValueError("Matrix cannot be empty.")

        if all(isinstance(i, (int, float)) for i in rows):
            self.entries = list(rows)
            self.m = 1
            self.n = len(rows)
        elif not all(isinstance(row, tuple) and len(row) == len(rows[0]) for row in rows):
            raise TypeError("Each row must be a tuple of equal length.")
        else:
            self.entries = [float(i) for j in rows for i in j]
            self.m = len(rows)
            self.n = len(rows[0])



    def __repr__(self) -> str:
        if not self.entries:
            return "[]"

        if hasattr(self, 'sig_digits') and self.sig_digits >= 0:
            formatted: List[str] = [f"{val:.{self.sig_digits}g}" for val in self.entries]
        else:
            formatted: List[str] = [f"{val}" for val in self.entries]

        min_val: float = min(self.entries)
        max_val: float = max(self.entries)
        range_val: float = max_val - min_val

        def get_color(value: float) -> str:
            if range_val == 0:
                return "\033[0m"
            normalized: float = (value - min_val) / range_val
            normalized = max(0.0, min(1.0, normalized))
            colors: List[int] = [82, 118, 226, 214, 196]
            color_idx: int = int(normalized * (len(colors) - 1))
            return f"\033[38;5;{colors[color_idx]}m"

        width: int = max(len(f) for f in formatted)
        reset: str = "\033[0m"
        rows_list: List[str] = []

        if len(self.entries) <= self.n:
            row_parts: List[str] = [
                f"{get_color(val)}{display:>{width}}{reset}" if getattr(self, 'use_color', False) else f"{display:>{width}}"
                for val, display in zip(self.entries, formatted)
            ]
            row: str = '  '.join(row_parts)
            return f"[ {row} ]"

        for j in range(0, len(self.entries), self.n):
            row_parts: List[str] = []
            for idx, val in enumerate(self.entries[j:j+self.n]):
                display: str = formatted[j + idx]
                if getattr(self, 'use_color', False):
                    row_parts.append(f"{get_color(val)}{display:>{width}}{reset}")
                else:
                    row_parts.append(f"{display:>{width}}")

            row: str = '  '.join(row_parts)
            if j == 0:
                rows_list.append(f"┌ {row} ┐")
            elif j + self.n >= len(self.entries):
                rows_list.append(f"└ {row} ┘")
            else:
                rows_list.append(f"│ {row} │")

        return '\n'.join(rows_list)



    def _to_tuple_form(self) -> List[Tuple[float, ...]]:
        """HELPER FUNCTION TO CONVERT FLATTENED DATA self.entries TO TUPLE FORMAT"""
        return Matrix.to_tuple_form(self.entries, self.n, self.m)
    @staticmethod
    def to_tuple_form(lst: List[float], n: int, m: int) -> List[Tuple[float, ...]]:
        """FUNCTION TO CONVERT FLATTENED DATA self.entries TO TUPLE FORMAT, ACCESSED EXTERNALLY"""
        return [tuple(lst[i*n:(i+1)*n]) for i in range(m)]

    from typing import Self

    @alias("ident", "IDENT", "I")
    @classmethod
    def identity(cls, n: int) -> Self:
        if n <= 0:
            raise ValueError(f"Provided matrix dimension (n={n}) must be greater than 0")
        
        entries: List[float] = [
            1.0 if i == j else 0.0
            for i in range(n)
            for j in range(n)
        ]
        return cls(*cls.to_tuple_form(entries, n, n))
    
    @alias("zero", "ZERO")
    @classmethod
    def zero_matrix(cls, n: int) -> Self:
        if n <= 0:
            raise ValueError(f"Provided matrix dimension (n={n}) must be greater than 0")
        return cls(*cls.to_tuple_form([0.0] * (n*n), n, n))

    @alias("rand", "RAND", "R")
    @classmethod
    def random(cls, n: int, m: int, low: float = 0, high: float = 1) -> Self:
        if n <= 0 or m <= 0:
            raise ValueError(f"Matrix dimensions must be positive (got {n}x{m})")
        if low > high:
            raise ValueError(f"Low bound {low} cannot be greater than high bound {high}")
        
        import random
        entries: List[float] = [random.uniform(low, high) for _ in range(n * m)]
        return cls(*cls.to_tuple_form(entries, n, m))
    
    @alias("T")
    @property
    def transpose(self) -> Self:
        """RETURN TRANSPOSED MATRIX"""
        transposed_entries: List[float] = []
        for j in range(self.n):
            for i in range(self.m):
                transposed_entries.append(self.entries[i*self.n + j])
        return Matrix(*self.to_tuple_form(transposed_entries, self.m, self.n))
    
    @alias("det")
    @property
    @validate_dimensions("square")
    def determinant(self) -> float:
        """RETURNS DETERMINANT OF MATRIX"""
        if self.n == 1:
            return self.entries[0]
        elif self.n == 2:
            return self.entries[0]*self.entries[3] - self.entries[1]*self.entries[2]
        
        if self.use_C:
            return cmat.mat_det(self.entries, self.n)
        
        """TO IMPLEMENT PYTHONIC VERSION, LAPLACE EXPANSION"""
        


    @validate_dimensions("elementwise")
    @performance_warning()
    def __add__(self, other: Self) -> Self:
        """ORDINARY MATRIX ADDITION"""
        if not self.force_C:
            summed_entries: List[float] = [i+j for i,j in zip(self.entries, other.entries)]
            return Matrix(*Matrix.to_tuple_form(summed_entries, self.n, self.m))
        
        C_entries: List[float] = cmat.mat_add(self.entries, other.entries, self.m, self.n)
        return Matrix(*self.to_tuple_form(C_entries, self.n, self.m))

    @validate_dimensions("elementwise")
    @performance_warning()
    def __sub__(self, other: Self) -> Self:
        """ORDINARY MATRIX SUBTRACTION"""
        if not self.force_C:
            subbed_entries: List[float] = [i-j for i,j in zip(self.entries, other.entries)]
            return Matrix(*Matrix.to_tuple_form(subbed_entries, self.n, self.m))
        
        C_entries: List[float] = cmat.mat_sub(self.entries, other.entries, self.m, self.n)
        return Matrix(*self.to_tuple_form(C_entries, self.n, self.m))


    @validate_dimensions("matmul")
    @performance_warning()
    def __mul__(self, other: Union[Self, float, int]) -> Union[Self, float]:
        """MATRIX MULTIPLICATION"""
        if isinstance(other, (float, int)):
            return self._smul(float(other))

        if not self.use_C:
            mult_entries: List[float] = []
            for i in range(self.m):
                for j in range(other.n):
                    val: float = sum(self.entries[i*self.n + k] * other.entries[k*other.n + j] for k in range(self.n))
                    mult_entries.append(val)
            return Matrix(*Matrix.to_tuple_form(mult_entries, other.n, self.m))

        C_entries: List[float] = cmat.mat_mul(self.entries, other.entries, self.m, self.n, other.n)
        
        if len(C_entries) == 1 and self.m == 1 and other.n == 1:
            return float(C_entries[0])
            
        return Matrix(*self.to_tuple_form(C_entries, other.n, self.m))
    

    @validate_dimensions("elementwise")
    @performance_warning()
    def __matmul__(self, other: Union[Self, float, int]) -> Self:
        """HADAMARD PRODUCT IMPLEMENTATION"""
        if isinstance(other, (float, int)):
            return self._smul(float(other))
        
        if not self.force_C:
            mult_entries: List[float] = [i*j for i,j in zip(self.entries, other.entries)]
            return Matrix(*Matrix.to_tuple_form(mult_entries, self.n, self.m))
            
        C_entries: List[float] = cmat.hadamard(self.entries, other.entries, self.m, self.n)
        return Matrix(*self.to_tuple_form(C_entries, self.n, self.m))

  
    
    def __rmul__(self, other: Union[float, int]) -> Union[Self, float]:
        """SCALAR MULTIPLICATION IS COMMUTATIVE."""
        return self * other
    def __rmatmul__(self, other: Union[float, int]) -> Self:
        """HADAMARD MULTPLICATION IS COMMUTATIVE."""
        return self @ other
    def _smul(self, other: float) -> Self:
        """SCALAR MATRIX MULTIPLICATION HELPER"""
        return Matrix(*Matrix.to_tuple_form([other*i for i in self.entries], self.n, self.m))

if __name__ == "__main__":
    print("--- Matrix Determinant Tests ---")

    # 1. Test 2x2 Case (Handled by your Python logic)
    # det = (1*4) - (2*3) = -2
    m2x2 = Matrix((1, 2), (3, 4))
    print(f"2x2 Matrix:\n{m2x2}")
    print(f"Determinant: {m2x2.det} | Expected: -2.0")
    print("-" * 30)

    # 2. Test 3x3 Case (This will trigger cmat.det if use_C is True)
    # Using a known matrix where det = 0
    m3x3_singular = Matrix(
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9)
    )
    print(f"3x3 Singular Matrix:\n{m3x3_singular}")
    print(f"Determinant: {m3x3_singular.det} | Expected: 0.0")
    print("-" * 30)

    # 3. Test Identity Matrix
    size = 4
    m_ident = Matrix.identity(size)
    print(f"{size}x{size} Identity Matrix:\n{m_ident}")
    print(f"Determinant: {m_ident.det} | Expected: 1.0")
    print("-" * 30)

    # 4. Test Error Handling (Non-square matrix)
    try:
        m_rect = Matrix((1, 2, 3), (4, 5, 6))
        print("Testing non-square determinant...")
        _ = m_rect.det
    except Exception as e:
        print(f"Caught expected error for non-square matrix: {e}")

    # 5. Scalar Multiplication & Det check
    # det(kA) = k^n * det(A)
    k = 2
    m_scaled = m2x2 * k
    print(f"\nScaled 2x2 (factor {k}):\n{m_scaled}")
    print(f"New Det: {m_scaled.det} | Expected (k^2 * -2): -8.0")