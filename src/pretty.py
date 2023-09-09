import shutil
from typing import Any, Callable, Optional

import numpy as np
from rich.console import Console

console = Console()


def print_row(
    row: dict[str, Any],
    show_header: bool = True,
    formats: Optional[dict[str, Callable[[Any], str]]] = None,
    widths: Optional[dict[str, float]] = None,
):
    if formats is None:
        formats = {}
    if widths is None:
        widths = {}
    for k, v in widths.items():
        assert 0 < v < 1, f"Widths should be between 0 and 1, got {v} for {k}"
    assert sum(widths.values()) <= 1, f"Sum of widths should be <= 1, got:\n{widths}"

    term_size = shutil.get_terminal_size((80, 20))

    claimed = sum(widths.values())
    remaining = 1 - claimed
    default_width = remaining / (len(row) - len(widths))
    widths = {k: widths.get(k, default_width) for k in row}

    def col_width(col: str):
        return int(np.round(widths[col] * term_size.columns))

    if show_header:
        header = [f"{k[:col_width(k) - 2]:<{col_width(k)}}" for k in row]
        console.print("".join(header), style="underline")
    row_str = ""
    for column, value in row.items():
        format = formats.get(column)
        if format is None:
            if isinstance(value, float):
                format = lambda x: f"{x:.3f}"
            else:
                format = str
        value_str = format(value)
        # Set the width of each column to 10 characters
        value_str = f"{value_str:<{col_width(column)}}"
        row_str += f"{value_str}"
    console.print(row_str)
