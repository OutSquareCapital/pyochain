"""Profile Python scripts with samply, and read the recorded profiles. Exploratory work."""

from __future__ import annotations

import gzip
import http.client
import signal
import subprocess
import sys
import urllib.parse
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import polars as pl

from pyochain import NONE, Iter, Some, option

if TYPE_CHECKING:
    from pathlib import Path


def profiler() -> None:  # ruff: ignore[undocumented-public-function]
    target = subprocess.Popen((sys.executable, "t.py"))
    p_args = ("samply", "record", "-p", str(target.pid))
    samply = subprocess.Popen(p_args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    _ = target.wait()
    samply.send_signal(signal.CTRL_BREAK_EVENT)
    with suppress(KeyboardInterrupt):
        _ = samply.wait()
    samply.kill()


def show(path: Path, top: int) -> None:  # ruff: ignore[undocumented-public-function]
    shared = pl.col("shared")
    profile = pl.read_json(
        gzip.decompress(path.read_bytes()), infer_schema_length=None
    ).lazy()
    frames = (
        profile
        .select(
            shared.struct.field("frameTable").struct.field("lib", "address", "func")
        )
        .explode("lib", "address", "func")
        .with_row_index("frame")
    )
    addresses = (
        frames
        .select("lib", "address")
        .drop_nulls()
        .filter(pl.col("address") >= 0)
        .unique(maintain_order=True)
        .collect()
    )
    body = (
        pl
        .concat(
            (
                profile.select(
                    pl
                    .col("libs")
                    .list.eval(
                        pl.concat_list(
                            pl.element().struct.field("debugName"),
                            pl.element().struct.field("breakpadId"),
                        )
                    )
                    .alias("memoryMap")
                ),
                addresses.lazy().select(
                    pl.concat_list("lib", "address").implode().implode().alias("stacks")
                ),
            ),
            how="horizontal",
        )
        .select(pl.concat_list(pl.struct("memoryMap", "stacks")).alias("jobs"))
        .collect()
        .write_ndjson()
    )
    server = subprocess.Popen(
        ("samply", "load", "--no-open", str(path)),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    symbols = _run_server(server, body)
    names = (
        frames
        .join(
            pl.concat(
                (
                    addresses.lazy(),
                    symbols
                    .lazy()
                    .select(
                        pl
                        .col("results")
                        .list.first()
                        .struct.field("stacks")
                        .list.first()
                        .alias("symbol")
                    )
                    .explode("symbol")
                    .select(
                        pl.col("symbol").struct.field("function").alias("outer"),
                        pl
                        .col("symbol")
                        .struct.field("inlines")
                        .list.first()
                        .struct.field("function")
                        .alias("inner"),
                    ),
                ),
                how="horizontal",
            ),
            on=("lib", "address"),
            how="left",
        )
        .join(
            profile
            .select(
                shared.struct
                .field("stringArray")
                .list.gather(shared.struct.field("funcTable").struct.field("name"))
                .alias("fallback")
            )
            .explode("fallback")
            .with_row_index("func")
            .with_columns(pl.col("func").cast(pl.Int64)),
            on="func",
        )
        .select(
            pl.col("frame").cast(pl.Int64),
            pl.coalesce("inner", "outer", "fallback").alias("inner"),
            pl.coalesce("outer", "fallback").alias("outer"),
        )
    )
    stacks = (
        profile
        .select(shared.struct.field("stackTable").struct.field("frame", "prefixOffset"))
        .explode("frame", "prefixOffset")
        .with_row_index("stack")
        .select(
            pl.col("stack").cast(pl.Int64),
            pl
            .when(pl.col("prefixOffset") != 0)
            .then(pl.col("stack").cast(pl.Int64) - pl.col("prefixOffset"))
            .alias("prefix"),
            "frame",
        )
        .join(names, on="frame")
        .collect()
    )
    samples = (
        profile
        .select(pl.col("threads"))
        .explode("threads")
        .select(pl.col("threads").struct.field("samples").struct.field("stack"))
        .filter(pl.col("stack").list.len() == pl.col("stack").list.len().max())
        .head(1)
        .explode("stack")
        .drop_nulls()
    )
    ancestors = (
        Iter
        .successors(
            Some(stacks.select(pl.col("stack").alias("leaf"), "prefix", "outer")),
            lambda level: (
                level
                .select("leaf", pl.col("prefix").alias("stack"))
                .drop_nulls()
                .join(stacks, on="stack")
                .select("leaf", "prefix", "outer")
                .pipe(lambda df: NONE if df.is_empty() else Some(df))
            ),
        )
        .map(lambda level: level.select("leaf", "outer"))
        .collect(pl.concat)
        .unique()
    )
    counts = (
        pl
        .concat((
            samples.join(stacks.lazy(), on="stack").select(
                pl.lit("self").alias("kind"), pl.col("inner").alias("function")
            ),
            samples.join(ancestors.lazy(), left_on="stack", right_on="leaf").select(
                pl.lit("inclusive").alias("kind"), pl.col("outer").alias("function")
            ),
        ))
        .group_by("kind", "function")
        .len("samples")
        .join(samples.select(pl.len().alias("total")), how="cross")
        .select(
            "kind",
            (pl.col("samples") / pl.col("total") * 100).round(1).alias("%"),
            "samples",
            "function",
        )
        .sort("kind", "samples", descending=True)
        .group_by("kind", maintain_order=True)
        .head(top)
        .collect()
    )
    with pl.Config(tbl_rows=-1, fmt_str_lengths=150, tbl_width_chars=250):
        print(counts)


def _run_server(server: subprocess.Popen[str], body: str) -> pl.DataFrame:
    try:
        url = (
            option(server.stdout)
            .map(lambda out: cast("str", out.readline()))
            .map(lambda s: s.split("symbolServer=")[1].strip())
            .map(urllib.parse.unquote)
            .map(urllib.parse.urlsplit)
            .expect("Failed to get symbol server URL from samply output")
        )
        client = http.client.HTTPConnection(url.netloc)
        client.request(
            "POST",
            f"{url.path}/symbolicate/v5",
            body,
            {"Content-Type": "application/json"},
        )
        symbols = pl.read_json(client.getresponse().read(), infer_schema_length=None)
        client.close()
    finally:
        server.kill()
    return symbols
