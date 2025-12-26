"""Check that all Rust iterator functions have a Python equivalent."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import polars as pl

import pyochain as pc

DATA = Path("scripts", "data")


ITERATOR_FN = {
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
}


def _iter_filter() -> pc.SetFrozen[str]:
    pure_rust = pc.SetFrozen.from_("copied", "cloned", "by_ref", "from_coroutine")
    """Methods that are not pertinent in the Python context."""
    equivalent: pc.SetFrozen[tuple[str, ...]] = pc.SetFrozen.from_(
        ("count", "length"),  # count is reserved for MutableSequence in Python
        (
            "is_sorted",
            "is_sorted_by_key",
            "is_sorted_by",
        ),  # all covered by is_sorted in Python
        (
            "from_",
            "into",
        ),  # from is a reserved word in Python, into is implicitely implemented with From trait in Rust
    )
    """Methods that have an equivalent Python counterpart."""
    py_stdlib = pc.SetFrozen.from_("filter_false")
    """Methods that exist in python stdlib but not in Rust."""

    return pure_rust.union(equivalent.iter().flatten()).union(py_stdlib)


def _decorated(fn: Callable[..., Any]) -> Callable[..., Any]:
    if isinstance(fn, (staticmethod, classmethod)):
        return fn.__func__  # type: ignore[return-value]
    return fn


def _with_source(fn_name: str, src: Literal["python", "rust"]) -> tuple[str, str]:
    return (src, fn_name)


def main(dtype: type, rust_fns: set[str], filters: pc.SetFrozen[str]) -> None:
    """Run the check and output the results to a ndjson file."""
    fn: pl.Expr = pl.col("fn")

    return (
        pc.Iter(dtype.mro())
        .map(lambda x: x.__dict__.values())
        .flatten()
        .filter(lambda x: callable(x) or isinstance(x, (staticmethod, classmethod)))
        .map(_decorated)
        .map(lambda x: _with_source(x.__name__, "python"))
        .chain(pc.Iter(rust_fns).map(lambda x: _with_source(x, "rust")))
        .into(lambda x: pl.LazyFrame(x, schema=["source", "fn"]))
        .filter(
            fn.is_unique().and_(
                fn.str.starts_with("_").not_().and_(fn.is_in(filters).not_())
            )
        )
        .sort(["source", "fn"])
        .sink_ndjson(DATA.joinpath(f"{dtype.__name__}_fns.ndjson"))
    )


if __name__ == "__main__":
    main(pc.Iter, ITERATOR_FN, _iter_filter())
