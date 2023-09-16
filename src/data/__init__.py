import importlib
from pathlib import Path
from typing import Union

from data.base import Data


def make(path: Union[str, Path], *args, **kwargs) -> Data:
    path = Path(path)
    name = path.stem
    name = ".".join(path.parts)
    module = importlib.import_module(name)
    return module.Data(*args, **kwargs)
