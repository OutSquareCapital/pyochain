# Contributing to pyochain

## Documentation

Each function/method should have a clear docstring, full type hints, with overloads and generics as needed.

This can be deconstructed into:

- A brief description of what the function does.
- A description of each argument in the Args section.
- A blank line.
- An example usage in a code block.
- After the expected return, a blank line.
- The triple backticks to close the code block.
- This can be repeated (explanation, code block, blank line, return, blank line) as needed for more complex functions.
- The closing triple quotes.

The weird end is due to the usage of docstring parsers that can parse examples in code blocks.

Example:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of what the function does.

    Args:
        param1 (int): Description of param1.
        param2 (str): Description of param2.
    Returns:
        bool: Description of the return value.
    
    Examples:
    ```python
    >>> # Example usage
    >>> my_function(5, "test")
    True

    ```
    """
    ...
```

## Architecture

The codebase is organized into several modules, each responsible for different aspects of the library. The use of leading underscores (`_`) for module names indicates that they are internal and not part of the public API. The public API is explicitly defined in `src/pyochain/__init__.py`.

- **`_core/`**: Contains the fundamental building blocks shared across all wrappers.
  - `_main.py`: Defines `CommonBase`, the abstract base class for all wrappers, which includes `pipe`, `into`, `apply`, and `inner` methods. It also contains base wrapper implementations like `IterWrapper` and `MappingWrapper`.
  - `_protocols.py`: Holds `typing.Protocol` definitions for structural type checking (e.g., `SupportsKeysAndGetItem`).
  - `_format.py`: Provides helper functions for pretty-printing and formatting wrapper contents.

- **`_iter/`**: Implements the `Iter` and `Seq` classes and their associated methods for lazy and eager iteration.
- **`_dict/`**: Implements the `Dict` class for chainable dictionary operations.
- **`_results/`**: Contains the `Result` and `Option` types for robust error and optional value handling.
  - `_result.py`: Defines `Result`, `Ok`, `Err`, and the `ResultUnwrapError` exception.
  - `_option.py`: Defines `Option`, `Some`, `NONE`, and the `OptionUnwrapError` exception.

### Structure of `_iter` and `_dict` Packages

These packages follow a mixin-based architecture to keep the code organized and maintainable.

- **Method Modules (`_<category>.py`)**: Each package is divided into modules that group related methods (e.g., `_filters.py`, `_maps.py`, `_joins.py`). Each of these modules contains a base class (e.g., `BaseFilter`, `MapDict`) that holds a set of methods.
- **Main Class (`_main.py`)**: The main public class (`Iter` or `Dict`) inherits from all the base classes defined in the method modules. This composition via multiple inheritance creates a single, powerful class with a rich API. This file also implements the core logic, such as constructors and methods that don't fit into a specific category.

This architecture allows for a clear separation of concerns, making the codebase easier to navigate and maintain, while still providing a flat and intuitive public API.

### Note on Mixins vs. Composition

The choice of using mixins (multiple inheritance) over composition (having instances of other classes as attributes) was made to provide a more seamless and intuitive API for users. This way, users can access all methods directly from the main class (`Iter`, `Dict`, etc.) without needing to navigate through multiple layers of objects (e.g., `my_iter.filtering.filter(...)`).

However, in some cases, composition is used to create namespaces for clarity. For example, `Iter.struct(...)` provides a way to apply `Dict` methods to an `Iter` of dictionaries, acting as a bridge between the two wrapper types.

## Setup

After cloning the repo, run:

```bash
uv sync --dev
```

## Testing

Before committing, ensure all tests pass and code quality checks are satisfied by running:

- **pydoclint** -> checks for missing or incomplete docstrings.
- **ruff** -> lints and formats the code.
- **ty** -> runs type-checking.
- **pytest** -> runs all tests on docstrings.

```bash
uv run pydoclint src/pyochain
uv run ruff check --fix src/pyochain
uv run ruff format src/pyochain
uv run ty check src/pyochain
uv run pytest --doctest-modules src/pyochain
```

### Internal code logic

Internal home-implementations of methods use "vanilla" python declarative iterations.

This can seem a bit conflictual with the public API purpose, but is done to minimize function call overhead for performance.
