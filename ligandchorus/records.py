"""Read the original sharded structural records and standalone NPZ inputs."""

import json
import zipfile
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from .features import load_record

try:
    from numpy._core.multiarray import _reconstruct
except ImportError:
    from numpy.core.multiarray import _reconstruct


def _reconstruct_numpy_array(*args):
    return _reconstruct(*args)


# Original archives record the NumPy 1.x import path. NumPy 2 moved it.
_reconstruct_numpy_array.__module__ = "numpy.core.multiarray"
_reconstruct_numpy_array.__name__ = _reconstruct_numpy_array.__qualname__ = "_reconstruct"
_NUMPY_SHARD_GLOBALS = [
    bytes,
    np.ndarray,
    np.dtype,
    _reconstruct,
    _reconstruct_numpy_array,
    *(type(np.dtype(name)) for name in ("bool", "float32", "int32", "int64")),
]


def manifest_entries(manifest):
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError(f"Expected an object at {manifest}:{line_number}")
                yield entry


def training_entries(manifest, limit):
    rows = (entry for entry in manifest_entries(manifest) if entry.get("split", "train") == "train")
    return list(islice(rows, limit))


def find_entry(manifest, key):
    matches = [entry for entry in manifest_entries(manifest) if entry.get("key") == key]
    if len(matches) != 1:
        raise ValueError(f"Expected one record for {key!r} in {manifest}, found {len(matches)}")
    return matches[0]


class RecordStore:
    """Reuse the last shard in manifest order while loading individual records."""

    def __init__(self, manifest, shard_root=None):
        self.manifest_root = manifest.parent
        self.shard_root = (shard_root or self.manifest_root).resolve()
        self.shard_path = None
        self.shard = None

    def read(self, entry):
        if "path" in entry:
            return load_record(self.manifest_root / entry["path"])
        if "shard" not in entry or type(entry.get("index")) is not int or entry["index"] < 0:
            raise ValueError("A record must provide an NPZ path or a shard and a nonnegative index")
        path = (self.shard_root / Path(entry["shard"])).resolve()
        if not path.is_relative_to(self.shard_root):
            raise ValueError(f"Shard is outside the configured data directory: {path}")
        if path != self.shard_path:
            if not path.is_file():
                raise FileNotFoundError(path)
            if not zipfile.is_zipfile(path):
                raise ValueError(f"Expected a ZIP-format PyTorch data shard: {path}")
            with torch.serialization.safe_globals(_NUMPY_SHARD_GLOBALS):
                self.shard = torch.load(path, map_location="cpu", weights_only=True)
            self.shard_path = path
        return self.shard[entry["index"]]
