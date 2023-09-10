import importlib
from pathlib import Path

from data.base import Data


def make(path: Path, *args, **kwargs) -> Data:
    src, *directories, _ = path.parts  # remove the file extension
    del src
    name = path.stem
    name = ".".join([*directories, name])
    module = importlib.import_module(name)
    return module.Data(*args, **kwargs)
