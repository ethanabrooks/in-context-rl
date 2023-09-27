import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
import torch
from braceexpand import braceexpand
from rich import print
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
from torchsnapshot import Snapshot
from tqdm import tqdm

import wandb
from pretty import Timer

ARTIFACTS_DIR = "artifacts"
NAME_FILE = "name.txt"
DIR = "replay_buffer"

dtype_mapping = {
    torch.bool: bool,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
}


@dataclass
class MetaInfo:
    device: torch.device
    dtype: torch.dtype
    shape: torch.Size


def get_meta_info(*meta_files: Path) -> Iterator[tuple[str, type, tuple[int, ...]]]:
    for meta_file in meta_files:
        meta_info = MetaInfo(**torch.load(meta_file))
        name = meta_file.stem.removesuffix(".meta")
        dtype_torch = meta_info.dtype
        dtype_np = dtype_mapping[dtype_torch]

        # For the shape, we only care about the last dimensions since structured arrays are 1D
        shape = tuple(meta_info.shape)
        yield name, dtype_np, shape


def load(directory: Path) -> np.ndarray:
    meta_files = list(directory.glob("*.meta.pt"))
    meta_info = list(get_meta_info(*meta_files))

    lens = [n for _, _, (n, _) in meta_info]
    try:
        [total_entries] = set(lens)
    except ValueError:
        raise RuntimeError(f"Multiple lengths found: {lens}")

    dtypes = [(n, t, d) for n, t, (_, *d) in meta_info]
    # Create an empty structured array
    structured_arr = np.empty(total_entries, dtype=np.dtype(dtypes))

    # Populate the structured array
    for name, dtype, shape in meta_info:
        memmap_file = directory / f"{name}.memmap"
        mmap_arr = np.memmap(memmap_file, dtype=dtype, mode="r", shape=shape)
        structured_arr[name] = mmap_arr

    return structured_arr


def compute_directory_checksum(dir_path: Path):
    """Compute a checksum for a directory and all its contents."""
    assert dir_path.exists()
    sha256_hash = hashlib.sha256()

    # Walk the directory and hash each file's contents
    for file in tqdm(sorted(dir_path.rglob("*")), desc="Hashing"):
        if file.is_file():  # Ensure we're only reading files
            with file.open("rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):  # Read in 4K chunks
                    sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def valid_checksum(data_dir: Path, dir_path: Path, verbose: bool = False):
    with Path("checksums.json").open("r") as f:
        checksums = json.load(f)
    key = str(dir_path.relative_to(data_dir))
    computed = compute_directory_checksum(dir_path)
    expected = checksums.get(key)
    if verbose:
        print("Computed:", computed)
        print("Expected:", expected)
    return computed == expected


def download(data_dir: Path, *artifact_patterns: str):
    api = wandb.Api()
    scratches_dir = data_dir / DIR
    artifacts_dir = data_dir / ARTIFACTS_DIR
    with Path("checksums.json").open("r") as f:
        checksums = json.load(f)

    def write_checksum(path: Path):
        key = str(path.relative_to(data_dir))
        checksum = compute_directory_checksum(path)
        checksums[key] = checksum

        with Path("checksums.json").open("w") as f:
            json.dump(checksums, f, indent=4, sort_keys=True)

    for pattern in artifact_patterns:
        buffers = []
        scratch_dir = scratches_dir / pattern
        if scratch_dir.exists():
            if valid_checksum(data_dir, scratch_dir, verbose=True):
                print(
                    f"Data stored in {scratch_dir} is valid. Skipping download/restoration."
                )
                continue
        shutil.rmtree(str(scratch_dir), ignore_errors=True)
        scratch_dir.mkdir(exist_ok=True, parents=True)

        for name in braceexpand(pattern):
            artifact_dir = artifacts_dir / name
            checksum_is_valid = valid_checksum(data_dir, artifact_dir, verbose=True)
            if artifact_dir.exists() and checksum_is_valid:
                print(f"Data stored in {artifact_dir} is valid. Skipping download.")
            else:
                shutil.rmtree(str(artifacts_dir / name), ignore_errors=True)
                artifact_dir.mkdir(exist_ok=True, parents=True)
                with open(artifact_dir / NAME_FILE, "w") as f:
                    f.write(name)
                artifact = api.artifact(name)
                while True:
                    try:
                        artifact.download(root=str(artifact_dir))
                        break
                    except requests.exceptions.ChunkedEncodingError as e:
                        print(e)
                        print("\nRetrying download...\n")
                write_checksum(artifact_dir)

            # load buffers
            run_buffers = {}
            for path in sorted((artifact_dir / "0").glob("[0-9]*")):
                replay_buffer = ReplayBuffer(
                    LazyMemmapStorage(max_size=0, scratch_dir=str(scratch_dir))
                )
                run_buffers[path.stem] = replay_buffer
            snapshot = Snapshot(path=str(artifact_dir))
            with Timer("Restoring snapshot"):
                snapshot.restore(run_buffers)
            buffers.extend(run_buffers.values())

        # merge buffers
        size = sum(len(buffer) for buffer in buffers)
        replay_buffer = ReplayBuffer(LazyMemmapStorage(size, scratch_dir=scratch_dir))
        for buffer in tqdm(buffers, desc="Merging buffers"):
            tensordict: TensorDict = buffer[:]
            done_mdp: torch.Tensor
            [done_mdp] = tensordict["done_mdp"].T
            last: torch.Tensor
            [*_, last] = done_mdp.nonzero()
            last = last.item()
            tensordict.set_at_("done", True, last)  # terminate last transition per task
            tensordict = tensordict[: last + 1]  # eliminate partial episodes
            replay_buffer.extend(tensordict)

        write_checksum(scratch_dir)


if __name__ == "__main__":
    import os

    from omegaconf import OmegaConf

    data_dir = Path(os.getenv("DATA_DIR"))
    assert data_dir.exists()

    artifacts = OmegaConf.load("artifacts.yml")
    download(
        data_dir,
        artifacts.point_env,
        artifacts.ant_dir,
        artifacts.cheetah_vel,
    )
