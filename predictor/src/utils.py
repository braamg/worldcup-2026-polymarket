"""Shared utilities: I/O helpers, disk cache, logging, and reproducibility."""
import functools
import hashlib
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and the hash environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def read_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)


# ---------------------------------------------------------------------------
# Disk-cache decorator
# ---------------------------------------------------------------------------

def disk_cache(cache_dir: str | Path):
    """
    Decorator factory: cache a function's return value to disk.

    The cache key is a SHA-256 hash of the function's qualified name plus its
    arguments.  Wrap any raw scrape / download function with this so the remote
    source is never hit more than once.

    Usage::

        @disk_cache("data/raw/example_source")
        def fetch_something(team: str, date: str) -> pd.DataFrame:
            ...
    """
    cache_path = Path(cache_dir)

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            cache_path.mkdir(parents=True, exist_ok=True)
            key = f"{fn.__module__}.{fn.__qualname__}:{args!r}:{sorted(kwargs.items())!r}"
            digest = hashlib.sha256(key.encode()).hexdigest()[:16]
            cache_file = cache_path / f"{fn.__name__}_{digest}.pkl"
            log = get_logger(__name__)

            if cache_file.exists():
                log.info("Cache hit: %s", cache_file.name)
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = fn(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            log.info("Cached → %s", cache_file.name)
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml, searching upward from this file's location if no path given."""
    if path is None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "config.yaml"
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError("config.yaml not found in any parent directory")
    with open(path) as f:
        return yaml.safe_load(f)
