from hjortMatrixHelper import CFunc
import shutil

def hjort__repr__(self) -> str:
    if self.m == 0 or self.n == 0:
        return "[]"
    def round_to_sig_figs(x, n):
        rounded = float(f"{x:.{n-1}e}")
        return f"{rounded:.{n}f}"

    def colorize(value: str, float_value: float, min_val: float, max_val: float, use_color: bool) -> str:
        if not use_color:
            return value
        range_val = max_val - min_val
        if range_val == 0:
            return value
        normalized = (float_value - min_val) / range_val
        normalized = max(0.0, min(1.0, normalized))
        colors = [82, 118, 226, 214, 196]
        color_idx = int(normalized * (len(colors) - 1))
        return f"\033[38;5;{colors[color_idx]}m{value}\033[0m"
    
    entries = CFunc.matrix_to_list(self._ptr)
    digits = self._flags.sig_digits
    use_color = self._flags.use_color
    limit_prints = self._flags.limit_prints

    min_val = CFunc.matrix_get_min(self._ptr)
    max_val = CFunc.matrix_get_max(self._ptr)
    
    output: str = ""
    for a,i in enumerate(entries):
        if limit_prints and a >= limit_prints - 1 and a != len(entries) - 1:
            if a >= limit_prints:
                continue


            sample = round_to_sig_figs(entries[0][0], digits)
            col_width = len(sample) + 1

            dot_row = ""
            for b in range(len(i)):

                if b >= limit_cols - 1 and b != len(i) - 1:
                    if b >= limit_cols:
                        continue

                dot = ".".center(col_width - 1)
                dot_row += dot + " "
            
            
            output += dot_row.rstrip() + "\n"
            continue

        row: str = ""
        sample = round_to_sig_figs(entries[0][0], digits)
        col_width = len(sample)
        term_width = shutil.get_terminal_size().columns
        max_cols_terminal = term_width // (col_width + 1)
        limit_cols = min(limit_prints, max_cols_terminal - 5) if limit_prints else max_cols_terminal - 5


        for b, j in enumerate(i):


            if limit_cols and b >= limit_cols - 1 and b != len(i) - 1:
                if b >= (limit_cols-1)+     1:  # NOTE Controls deprecated column dot count
                    continue
                dot = ".".center(col_width)
                row += dot + " "
                continue


            formatted = round_to_sig_figs(j, digits)
            row += colorize(formatted, j, min_val, max_val, use_color) + " "
        
        row_content = row.rstrip()

        if a == 0:
            left, right = "⎡", "⎤"
        elif a == len(entries) - 1:
            left, right = "⎣", "⎦"
        else:
            left, right = "⎢", "⎥"

        output += f"{left} {row_content} {right}\n"

    return output