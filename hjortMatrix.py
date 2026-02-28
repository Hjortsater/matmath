from hjortMatrixHelper import CFunc

"""

Simple linear algebra class written for fun, for experience, for ease of use and hopefully actual implementations down the line!
This version of the code boasts a more prominent C-backend, instead of only off-loading heavy operations via ctypes.
    *This way multithreading should scale
    *This way type-conversion overhead should be reduced
    
Written by Erik Hjortsäter February 27th 2026.

"""





class Matrix:

    ### DUNDER METHODS (and then some)

    class _Flags:
        def __init__(self, **kwargs):
            """Constructor for the flags helper"""
            self.sig_digits: int = kwargs.get("sig_digits", 3)
            self.use_color: bool = kwargs.get("use_color", True)
            self.multithreaded: bool = kwargs.get("multithreaded", True)

    def __init__(self, *rows, **flags) -> None:
        """Pythonic constructor, slow but Python-native"""
        self._flags = self._Flags(**flags)
        flat = rows

        if not len(rows):
            raise ValueError("Matrix cannot be empty.")

        # NOTE 1D vector input
        if all(isinstance(i, (int, float)) for i in rows):
            m, n = 1, len(rows)
            flat = rows

        # NOTE 2D matrix input (can still result in vector)
        elif all(isinstance(r, (list, tuple)) for r in rows):
            m = len(rows)
            n = len(rows[0])

            if any(len(r) != n for r in rows):
                raise ValueError("Rows must be same length.")

            flat = [val for r in rows for val in r]

        else:
            raise TypeError("Invalid constructor input.")
        
        import array
        buffer = array.array('d', flat)
        ptr = CFunc.matrix_create_from_buffer(buffer, m, n)
        
        if not ptr:
            raise MemoryError("C backend allocation failed")
        
        self._ptr = ptr

      
    @classmethod
    def __init__C_native(cls, ptr, **flags):
        """
        C-idiomatic constructor, takes a C pointer and optional flags.
        """
        if not ptr:
            raise MemoryError("Null pointer from C backend.")

        obj = cls.__new__(cls)      # bypass Python __init__
        obj._ptr = ptr
        obj._flags = cls._Flags(**flags)
        return obj

    __slots__ = ("_ptr", "_flags")

    def __del__(self) -> None:
        """Object destuctor, calls back-end freeing"""
        if hasattr(self, "_ptr") and self._ptr:
                CFunc.matrix_free(self._ptr)

    def __repr__(self) -> str:
        """Object representation when printed. Considers flags"""
        import sys
        if self.m == 0 or self.n == 0:
            return "[]"

        entries = [
            [CFunc.matrix_get(self._ptr, i, j) for j in range(self.n)]
            for i in range(self.m)
        ]

        digits = self._flags.sig_digits
        use_color = self._flags.use_color

        formatted = [
            [f"{val:.{digits}f}" for val in row]
            for row in entries
        ]

        col_widths = [
            max(len(formatted[i][j]) for i in range(self.m))
            for j in range(self.n)
        ]

        flat_vals = [val for row in entries for val in row]
        smallest = min(flat_vals)
        largest = max(flat_vals)
        diff = largest - smallest if largest != smallest else 1.0

        max_abs = max(abs(smallest), abs(largest), 1.0)
        eps = sys.float_info.epsilon * max_abs * 8

        gradient = [46, 82, 118, 154, 190, 226, 220, 214, 208, 202, 196]

        def colorize(val: float, text: str) -> str:
            if not use_color:
                return text
            if abs(val) <= eps:
                return text
            normalized = (val - smallest) / diff
            normalized = max(0.0, min(1.0, normalized))
            idx = int(normalized * (len(gradient) - 1))
            return f"\033[38;5;{gradient[idx]}m{text}\033[0m"

        rows = []
        for i in range(self.m):
            parts = []
            for j in range(self.n):
                raw_text = formatted[i][j].rjust(col_widths[j])
                parts.append(colorize(entries[i][j], raw_text))
            row_str = "  ".join(parts)
            if i == 0:
                rows.append(f"┌ {row_str} ┐")
            elif i == self.m - 1:
                rows.append(f"└ {row_str} ┘")
            else:
                rows.append(f"│ {row_str} │")

        return "\n".join(rows)
    
    ### DUNDER OPERATIONS

    def __add__(self, other):
        """Add two matrices using the C backend and return a new Matrix."""
        if not isinstance(other, Matrix):
            raise NotImplementedError

        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for addition.")

        new_ptr = CFunc.matrix_add(self._ptr, other._ptr, int(self._flags.multithreaded))
        if not new_ptr:
            raise MemoryError("C backend failed to allocate result matrix.")

        return Matrix.__init__C_native(
            new_ptr,
            sig_digits=self._flags.sig_digits,
            use_color=self._flags.use_color,
            multithreaded=self._flags.multithreaded
        )
    
    def __sub__(self, other):
        """Subtract two matrices using the C backend and return a new Matrix."""
        if not isinstance(other, Matrix):
            raise NotImplementedError

        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for addition.")

        new_ptr = CFunc.matrix_sub(self._ptr, other._ptr, int(self._flags.multithreaded))
        if not new_ptr:
            raise MemoryError("C backend failed to allocate result matrix.")

        return Matrix.__init__C_native(
            new_ptr,
            sig_digits=self._flags.sig_digits,
            use_color=self._flags.use_color,
            multithreaded=self._flags.multithreaded
        )
    
    def __mul__(self, other):
        """Subtract two matrices using the C backend and return a new Matrix."""
        if not isinstance(other, Matrix):
            raise NotImplementedError

        if self.n != other.m:
            raise ValueError("Matrix dimensions must match for addition.")

        new_ptr = CFunc.matrix_mul(self._ptr, other._ptr, int(self._flags.multithreaded))
        if not new_ptr:
            raise MemoryError("C backend failed to allocate result matrix.")
        
        return Matrix.__init__C_native(
            new_ptr,
            sig_digits=self._flags.sig_digits,
            use_color=self._flags.use_color,
            multithreaded=self._flags.multithreaded
        )
        

    ### PROPERTIES

    @property
    def m(self):
        return CFunc.matrix_rows(self._ptr)

    @property
    def n(self):
        return CFunc.matrix_cols(self._ptr)


    ### MISCELLANEOUS

    @classmethod
    def random(cls, m, n, min_val=0.0, max_val=1.0, **flags):
        """Create a random matrix using the C backend"""
        ptr = CFunc.matrix_create(m, n)
        if not ptr:
            raise MemoryError("Failed to allocate matrix in C backend.")
        
        CFunc.matrix_fill_random(ptr, min_val, max_val)

        return cls.__init__C_native(ptr, **flags)