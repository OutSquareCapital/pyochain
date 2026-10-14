"""Quick script to fetch and check benchmark data from original sortedcontainers repo.

TODO: use this to compare with pyochain results.
"""

import polars as pl

URL = "https://raw.githubusercontent.com/grantjenks/python-sortedcontainers/master/tests/results_sortedlist.txt"
SCHEMA = {
    "operation": pl.String,
    "implementation": pl.String,
    "n": pl.Int64,
    "min_s": pl.Float64,
    "max_s": pl.Float64,
    "median_s": pl.Float64,
    "mean_s": pl.Float64,
}


def run() -> None:
    """Fetch and display benchmark data from original sortedcontainers repo."""
    on_cols = ("operation", "n")
    median = pl.col("median_s")
    implementation = pl.col("implementation")
    return (
        pl
        .scan_csv(URL, separator=" ", has_header=False, schema=SCHEMA)
        .pipe(
            lambda df: df.join(
                df.filter(implementation.eq("SortedList")).select(
                    *on_cols, median.alias("baseline_s")
                ),
                on=on_cols,
                how="left",
            )
        )
        .select(
            *on_cols,
            implementation,
            median.alias("time_s"),
            median.truediv("baseline_s").round(2).alias("ratio"),
        )
        .sort("ratio", *on_cols, implementation)
        .collect()
        .show(-1)
    )


run()
