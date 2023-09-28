import shutil
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Callable, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, ProgressColumn, Task, TextColumn
from rich.text import Text

console = Console()


class Table:
    def __init__(
        self,
        formats: Optional[dict[str, Callable[[Any], str]]] = None,
    ):
        self.formats = {} if formats is None else formats
        self.term_size = shutil.get_terminal_size((80, 20))
        self.width = None

    @property
    def col_width(self) -> int:
        assert self.width is not None
        return int(np.round(self.width * self.term_size.columns))

    def print_header(self, row: dict[str, Any]):
        self.width = 1 / (len(row))
        header = [f"{k[:self.col_width - 2]:<{self.col_width}}" for k in row]
        console.print("".join(header), style="underline")

    def print_row(self, row: dict[str, Any]):
        row = dict(list(row.items())[:10])
        formats = self.formats

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
            value_str = f"{value_str:<{self.col_width}}"
            row_str += f"{value_str}"
        console.print(row_str)


def render_eval_metrics(
    *numbers: float, max_num: float, width: int = 10, length: int = 10
):
    if len(numbers) > length:
        subarrays = np.array_split(numbers, length)
        # Compute the mean of each subarray
        numbers = [subarray.mean() for subarray in subarrays]
    bar_elements = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    for num in numbers:
        num = min(num, max_num)
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
        yield f"{num:<6} {bar}{padding}▏"


class TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: Task) -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        delta = timedelta(seconds=elapsed)
        return Text(str(delta), style="progress.elapsed")


@contextmanager
def Timer(desc: Optional[str] = None):
    columns = [TimeElapsedColumn()]
    if desc:
        columns.append(TextColumn(f"[progress.description]{desc}"))
    with Progress(*columns) as progress:
        progress.add_task(desc, total=None)
        yield


# whitelist
TimeElapsedColumn.render
