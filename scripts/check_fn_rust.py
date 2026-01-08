"""Check that all Rust iterator functions have a Python equivalent."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import polars as pl

import pyochain as pc

DATA = Path("scripts", "data")


def _iterators_fn() -> pc.Set[str]:
    return pc.Set(
        (
            "advance_by",
            "all",
            "any",
            "array_chunks",
            "by_ref",
            "chain",
            "cloned",
            "cmp",
            "cmp_by",
            "collect",
            "collect_into",
            "copied",
            "count",
            "cycle",
            "enumerate",
            "eq",
            "eq_by",
            "filter",
            "filter_map",
            "find",
            "find_map",
            "flat_map",
            "flatten",
            "fold",
            "for_each",
            "fuse",
            "ge",
            "gt",
            "inspect",
            "intersperse",
            "intersperse_with",
            "is_partitioned",
            "is_sorted",
            "is_sorted_by",
            "is_sorted_by_key",
            "last",
            "le",
            "lt",
            "map",
            "map_while",
            "map_windows",
            "max",
            "max_by",
            "max_by_key",
            "min",
            "min_by",
            "min_by_key",
            "ne",
            "next_chunk",
            "nth",
            "partial_cmp",
            "partial_cmp_by",
            "partition",
            "partition_in_place",
            "peekable",
            "position",
            "product",
            "reduce",
            "rev",
            "rposition",
            "scan",
            "size_hint",
            "skip",
            "skip_while",
            "step_by",
            "sum",
            "take",
            "take_while",
            "try_collect",
            "try_find",
            "try_fold",
            "try_for_each",
            "try_reduce",
            "unzip",
            "zip",
            "empty",
            "from_coroutine",
            "from_fn",
            "once",
            "once_with",
            "repeat",
            "repeat_n",
            "repeat_with",
            "successors",
            "next",
        )
    )


def _iter_filter() -> pc.Set[str]:
    return pc.Set(
        (
            "ok_or",
            "ok_or_else",
            "then_some",
            "then",
            # Implemented via boolean in Rust
            "copied",
            "cloned",
            "by_ref",
            "from_coroutine",
            # not pertinent in Python
            "count",
            "length",  # count is reserved for MutableSequence in Python
            "is_sorted",
            "is_sorted_by_key",
            "is_sorted_by",
            # all covered by is_sorted in Python
            "into",  # into is implicitely implemented with From trait in Rust
            "filter_false",  # filter_false is in itertools in Python
            "iter",  # inerhited from base class in Python
            "max_by_key",  # already covered by max_by
            "min_by_key",  # already covered by min_by
            "new",  # Not present in the trait in Rust
        )
    )


def _decorated(fn: Callable[..., Any]) -> Callable[..., Any]:
    if isinstance(fn, (staticmethod, classmethod)):
        return fn.__func__  # type: ignore[return-value]
    return fn


def _with_source(fn_name: str, src: Literal["python", "rust"]) -> tuple[str, str]:
    return (src, fn_name)


def main(dtype: type, rust_fns: pc.Set[str], filters: pc.Set[str]) -> None:
    """Run the check and output the results to a ndjson file."""
    fn: pl.Expr = pl.col("fn")

    return (
        pc.Iter(dtype.mro())
        .flat_map(lambda x: x.__dict__.values())
        .filter(lambda x: callable(x) or isinstance(x, (staticmethod, classmethod)))
        .map(_decorated)
        .map(lambda x: _with_source(x.__name__, "python"))
        .chain(rust_fns.iter().map(lambda x: _with_source(x, "rust")))
        .into(lambda x: pl.LazyFrame(x, schema=["source", "fn"]))
        .filter(
            fn.is_unique().and_(
                fn.str.starts_with("_")
                .not_()
                .and_(fn.is_in(filters).not_().and_(fn.str.ends_with("_star").not_()))
            )
        )
        .sort(["source", "fn"])
        .sink_ndjson(DATA.joinpath(f"{dtype.__name__}_fns.ndjson"))
    )


if __name__ == "__main__":
    main(pc.Iter, _iterators_fn(), _iter_filter())
