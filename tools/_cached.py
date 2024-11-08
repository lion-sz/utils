from typing import Callable
import sys
import functools
from datetime import datetime, timedelta

import polars as pl

def cached(name: str, retention_hours: int = 0) -> Callable:
    """Cache a function result.
    Args:
        name (str): The name to use for the cache file.
        retention_hours (int): The number of hours that a file is cached for.
    """
    return lambda f: _cached(f, name, retention_hours)


def _cached(func, name: str, retention_hours: int):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        config = sys.modules["app_template"].config
        if config is None:
            raise ValueError("Cannot use `cached` without `app`.")
        if not config.use_cache:
            return func(*args, **kwargs)
        # Check if a cached file exists.
        cache_path = config.cache_path
        if not cache_path.exists():
            cache_path.mkdir()
        cache_file = None
        candidates = sorted(cache_path.glob(f"{name}_*.parquet"), reverse=True)
        # Go through the candidates and filter out too old ones.
        retention = timedelta(hours=retention_hours)
        for cand in candidates:
            cand_ts = int(cand.name.split("_")[-1].replace(".parquet", ""))
            cand_ts = datetime.fromtimestamp(cand_ts)
            if (datetime.now() - cand_ts) > retention:
                cand.unlink()
            # The first is the cache file.
            elif cache_file is None:
                cache_file = cand
        if cache_file is None:
            ts = int(datetime.now().timestamp())
            cache_file = cache_path / f"{name}_{ts}.parquet"
            res = func(*args, **kwargs)
            res.write_parquet(cache_file)
            return res
        else:
            return pl.read_parquet(cache_file)

    if inner.__doc__ is not None:
        parts = inner.__doc__.split("\n", maxsplit=1)
        header = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        note = (
            f"\n{4*' '}Note:\n"
            f"{8*' '}This function is cached with a retention policy of "
            f"{retention_hours} hours.\n"
        )
        inner.__doc__ = "\n".join([header, note, body])
    return inner
