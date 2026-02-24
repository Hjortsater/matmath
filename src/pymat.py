import cmat
from typing import Self
from customdecorators import alias

class Matrix():
    """
    CUSTOM MATRIX CLASS IN PYTHON WITH A C BACKEND FOR EXPENSIVE COMPUTATIONS.

    WRITTEN FOR FUN, NOT FOR SPEED!
    """

    def __init__(self, *rows) -> None:
        """CONSTRUCTOR FOR MATRIX"""
        self.entries: list = []
        self.m: int = 0  # Matrix height
        self.n: int = 0  # Matrix width
        
        self.use_C: bool = True
        self.use_color: bool = True
        self.sig_digits: int = 2


        if not rows:
            raise ValueError("Matrix cannot be empty.")

        if all(isinstance(i, (int, float)) for i in rows):
            self.entries = list(rows)
            self.m = 1            
            self.n = len(rows)    
        elif not all(isinstance(row, tuple) and len(row) == len(rows[0]) for row in rows):
            raise TypeError("Each row must be a tuple of equal length.")
        else:
            self.entries = [i for j in rows for i in j]
            self.m = len(rows)
            self.n = len(rows[0])

    def __repr__(self) -> str:
        if not self.entries:
            return "[]"
        
        if hasattr(self, 'sig_digits') and self.sig_digits >= 0:
            formatted = [f"{val:.{self.sig_digits}g}" for val in self.entries]
        else:
            formatted = [f"{val}" for val in self.entries]
        
        min_val = min(self.entries)
        max_val = max(self.entries)
        range_val = max_val - min_val
        
        def get_color(value):
            if range_val == 0 or value == 0:
                return "\033[0m"
            
            normalized = (value - min_val) / range_val
            
            colors = [82, 118, 226, 214, 196]
            color_idx = int(normalized * (len(colors) - 1))
            return f"\033[38;5;{colors[color_idx]}m"
        
        width = max(len(f) for f in formatted)
        reset = "\033[0m"
        rows = []
        
        for j in range(0, len(self.entries), self.n):
            row_parts = []
            for idx, val in enumerate(self.entries[j:j+self.n]):
                display = formatted[j + idx]
                if self.use_color:
                    row_parts.append(f"{get_color(val)}{display:>{width}}{reset}")
                else:
                    row_parts.append(f"{display:>{width}}")
            
            row = '  '.join(row_parts)
            
            if j == 0:
                rows.append(f"┌ {row} ┐")
            elif j + self.n >= len(self.entries):
                rows.append(f"└ {row} ┘")
            else:
                rows.append(f"│ {row} │")
        
        return '\n'.join(rows)

    def _to_tuple_form(self) -> Self:
        """HELPER FUNCTION TO CONVERT FLATTENED DATA self.entries TO TUPLE FORMAT"""
        return Matrix.to_tuple_form(self.entries, self.n, self.m)
    @staticmethod
    def to_tuple_form(lst: list, n: int, m: int) -> Self:
        """FUNCTION TO CONVERT FLATTENED DATA self.entries TO TUPLE FORMAT, ACCESSED EXTERNALLY"""
        return [tuple(lst[i*n:(i+1)*n]) for i in range(m)]

    from typing import Self

    @alias("ident", "IDENT", "I")
    @classmethod
    def identity(cls, n: int) -> Self:
        if n <= 0:
            raise ValueError(f"Provided matrix dimension (n={n}) must be greater than 0")
        
        entries = [
            1 if i == j else 0
            for i in range(n)
            for j in range(n)
        ]
        return cls(*cls.to_tuple_form(entries, n, n))
    
    @alias("zero", "ZERO")
    @classmethod
    def zero_matrix(cls, n: int) -> Self:
        if n <= 0:
            raise ValueError(f"Provided matrix dimension (n={n}) must be greater than 0")
        return cls(*cls.to_tuple_form([0] * (n*n), n, n))

    @alias("rand", "RAND", "R")
    @classmethod
    def random(cls, n: int, m: int, low: float = 0, high: float = 1) -> Self:
        if n <= 0 or m <= 0:
            raise ValueError(f"Matrix dimensions must be positive (got {n}x{m})")
        if low > high:
            raise ValueError(f"Low bound {low} cannot be greater than high bound {high}")
        
        import random
        entries = [random.uniform(low, high) for _ in range(n * m)]
        return cls(*cls.to_tuple_form(entries, n, m))

    def __add__(self, other: Self) -> Self:
        """ORDINARY MATRIX ADDITION"""
        if not type(other) == type(self):
            raise TypeError("Matrix addition failed: mismatched types.")
        if self.n != other.n or self.m != other.m:
            raise TypeError("Matrix addition failed: mismatched dimensions.")
        if not self.use_C:
            summed_entries: list = [i+j for i,j in zip(self.entries, other.entries)]
            return Matrix(*Matrix.to_tuple_form(summed_entries, self.n, self.m))
        
        C_entries = cmat.mat_add(self.entries, other.entries, self.m, self.n)
        return Matrix(*self.to_tuple_form(C_entries, self.n, self.m))

    def __sub__(self, other: Self) -> Self:
        """ORDINARY MATRIX SUBTRACTION"""
        if not type(other) == type(self):
            raise TypeError("Matrix subtraction failed: mismatched types.")
        if self.n != other.n or self.m != other.m:
            raise TypeError("Matrix subtraction failed: mismatched dimensions.")
        if not self.use_C:
            subbed_entries: list = [i-j for i,j in zip(self.entries, other.entries)]
            return Matrix(*Matrix.to_tuple_form(subbed_entries, self.n, self.m))
        
        C_entries = cmat.mat_sub(self.entries, other.entries, self.m, self.n)
        return Matrix(*self.to_tuple_form(C_entries, self.n, self.m))


    def __mul__(self, other: Self) -> Self:
        """MATRIX MULTIPLICATION"""
        if type(other) in (float, int):
            return self._smul(other)
        if type(other) == type(self):
            pass
            
    
    def __matmul__(self, other: Self) -> Self:
        """HADAMARD PRODUCT IMPLEMENTATION"""
        if type(other) in (float, int):
            return self._smul(other)
        if type(other) == type(self):
            if self.n != other.n or self.m != other.m:
                raise TypeError("Matrix multiplication failed: mismatched dimensions.")
            if not self.use_C:
                mult_entries: list = [i*j for i,j in zip(self.entries, other.entries)]
                return Matrix(*Matrix.to_tuple_form(mult_entries, self.n, self.m))
            C_entries = cmat.hadamard(self.entries, other.entries, self.m, self.n)
            return Matrix(*self.to_tuple_form(C_entries, self.n, self.m))

  
    
    def __rmul__(self, other) -> Self:
        """SCALAR MULTIPLICATION IS COMMUTATIVE."""
        return self * other
    def __rmatmul__(self, other) -> Self:
        """HADAMARD MULTPLICATION IS COMMUTATIVE."""
        return self @ other
    def _smul(self, other: Self) -> Self:
        """SCALAR MATRIX MULTIPLICATION HELPER"""
        if type(other) in (float, int):
            return Matrix(*Matrix.to_tuple_form([other*i for i in self.entries], self.n, self.m))


if __name__ == "__main__":
    A = Matrix.I(5)
    B = Matrix.R(5,5,0,10)

    print(A@B)