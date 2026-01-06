# Cookbook

This cookbook provides practical examples of how to use the `pyochain` library for various data manipulation tasks in Python.
Each example demonstrates a specific use case, showcasing the power and flexibility of `pyochain` for functional programming and data processing.

## Combining Option, Result and Iterators in Data Pipelines

Classes have been designed to work seamlessly together, enabling complex data processing pipelines with clear error handling.

**Note**: We return pc.Ok(None) for simplicity.

```python
>>> import polars as pl
>>> import pyochain as pc
>>>
>>> def safe_parse_int(s: str) -> pc.Result[int, str]:
...     try:
...         return pc.Ok(int(s))
...     except ValueError:
...         return pc.Err(f"Invalid integer: {s}")
>>>
>>> def _run_ok(lf: pl.LazyFrame) -> pc.Result[pl.LazyFrame, str]:
...     """Collect and run the pipeline."""
...     try:
...         return pc.Ok(lf.filter(pl.col("value").gt(15)))
...     except (pl.exceptions.ComputeError) as e:
...         return pc.Err(f"Failed to write to file: {e}")
>>>
>>> data = ["10", "20", "foo", "30", "bar"]
>>> results = (
...    pc.Iter(data)
...    .map(safe_parse_int)  # Parse each string safely
...    .filter_map(lambda r: r.ok())  # Keep only successful parses
...    .enumerate()  # Add indices
...    .collect()  # Materialize the results
...    .inspect(
...        lambda seq: print(f"Parsed integers: {seq}") # Log parsed integers
...    )
...    .into(pl.LazyFrame, schema=["index", "value"])  # Pass to Polars LazyFrame
...    .pipe(_run_ok)  # Run the pipeline
...    .map_err(lambda e: print(f"Error: {e}"))  # Print error message
...    .map(lambda _: None)
... )
Parsed integers: Seq((0, 10), (1, 20), (2, 30))
>>> results
Ok(None)

```

### Determining All Public Methods of a Class

Below is an example of using pyochain to:

**1.** extract all public methods of a class.
**2.** enumerate them
**3.** sort them by name
**4.** convert the first three into a dictionary.

```python
>>> import pyochain as pc
>>> 
>>> def get_public_methods(cls: type) -> dict[int, str]:
...     return (
...         pc.Iter(cls.mro())
...         .flat_map(lambda x: x.__dict__.values())
...         .filter(lambda f: callable(f) and not f.__name__.startswith("_"))
...         .map(lambda f: f.__name__)
...         .enumerate()
...         .sort(key=lambda pair: pair[1])
...         .iter()
...         .take(3)
...         .collect(dict)
...     )
>>>
>>> get_public_methods(pc.Iter)
{25: 'accumulate', 68: 'adjacent', 96: 'all'}


```

For comparison, here's the equivalent using pure Python:

```python
>>> import itertools
>>> import pyochain as pc
>>>
>>> def get_public_methods_pure(cls: type) -> dict[int, str]:
...     return dict(
...         itertools.islice(
...             sorted(
...                 enumerate(
...                     f.__name__
...                     for f in itertools.chain.from_iterable(
...                         map(lambda x: x.__dict__.values(), cls.mro())
...                     )
...                     if callable(f) and not f.__name__.startswith("_")
...                 ),
...                 key=lambda pair: pair[1],
...             ),
...             3,
...         )
...     )
>>>
>>> get_public_methods_pure(pc.Iter)
{25: 'accumulate', 68: 'adjacent', 96: 'all'}

```
