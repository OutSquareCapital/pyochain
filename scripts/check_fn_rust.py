"""Check that all Rust iterator functions have a Python equivalent."""

from pathlib import Path
from typing import Literal

import polars as pl

import pyochain as pc

DATA = Path("scripts", "data")

RUST_FN = (
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

PURE_RUST = {"copied", "cloned", "by_ref", "from_coroutine"}
"""Methods that are not pertinent in the Python context."""
PURE_PY = {
    "collect_mut"  # don't need in Rust cause collect can use watever collection
}
"""Methods that are not pertinent in the Rust context."""
EQUIVALENT = {
    ("count", "length"),  # count is reserved for MutableMappings in Python
    (
        "is_sorted",
        "is_sorted_by_key",
        "is_sorted_by",
    ),  # all covered by is_sorted in Python
    (
        "from_",
        "into",
    ),  # from is a reserved word in Python, into is implicitely implemented with From trait in Rust
}
"""Methods that have an equivalent Python counterpart."""
PY_STDLIB = {"filter_false"}
"""Methods that exist in python stdlib but not in Rust."""

ALL_FILTERS = (
    PURE_RUST.union(PURE_PY).union(pc.Iter(EQUIVALENT).flatten()).union(PY_STDLIB)
)


def _with_source(fn_name: str, src: Literal["python", "rust"]) -> tuple[str, str]:
    return (src, fn_name)


def main() -> None:
    """Run the check and output the results to a ndjson file."""
    fn: pl.Expr = pl.col("fn")

    return (
        pc.Iter(pc.Iter.mro())
        .map(lambda x: x.__dict__.values())
        .flatten()
        .filter(callable)
        .map(lambda x: _with_source(x.__name__, "python"))
        .chain(pc.Iter(RUST_FN).map(lambda x: _with_source(x, "rust")))
        .into(lambda x: pl.LazyFrame(x, schema=["source", "fn"]))
        .filter(
            fn.is_unique().and_(
                fn.str.starts_with("_").not_().and_(fn.is_in(ALL_FILTERS).not_())
            )
        )
        .sort(["source", "fn"])
        .sink_ndjson(DATA.joinpath("iter_fn_sources.ndjson"))
    )


if __name__ == "__main__":
    main()
