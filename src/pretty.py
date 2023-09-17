import shutil
from typing import Any, Callable, Optional

import numpy as np
from rich.console import Console

console = Console()


def print_row(
    row: dict[str, Any],
    show_header: bool = True,
    formats: Optional[dict[str, Callable[[Any], str]]] = None,
):
    row = dict(list(row.items())[:10])
    if formats is None:
        formats = {}

    term_size = shutil.get_terminal_size((80, 20))

    width = 1 / (len(row))

    def col_width(col: str):
        return int(np.round(width * term_size.columns))

    if show_header:
        header = [f"{k[:col_width(k) - 2]:<{col_width(k)}}" for k in row]
        console.print("".join(header), style="underline")
    row_str = ""
    for column, value in row.items():
        format = formats.get(column)
        if format is None:
            if isinstance(value, float):
                format = lambda x: "{:.{n}g}".format(x, n=3)
            else:
                format = str
        value_str = format(value)
        # Set the width of each column to 10 characters
        value_str = f"{value_str:<{col_width(column)}}"
        row_str += f"{value_str}"
    console.print(row_str)


def render_graph(*numbers: float, max_num: float, width: int = 10, length: int = 10):
    if len(numbers) > length:
        subarrays = np.array_split(numbers, length)
        # Compute the mean of each subarray
        numbers = [subarray.mean() for subarray in subarrays]
    bar_elements = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    for num in numbers:
        assert num <= max_num
        ratio = num / max_num
        full_blocks = int(ratio * width)
        fraction = ratio * width - full_blocks
        bar = full_blocks * "█"
        partial_block = round(fraction * (len(bar_elements) - 1))
        if num < max_num:
            bar += bar_elements[partial_block]
        padding = width - len(bar)
        padding = " " * padding
        num = round(num, 1)
        yield f"{num:<4} {bar}{padding}▏"
