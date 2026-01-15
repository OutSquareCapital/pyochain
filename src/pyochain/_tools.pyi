from collections.abc import Callable, Iterator

from pyochain import Option, Result

def try_find[T, E](
    data: Iterator[T], predicate: Callable[[T], Result[bool, E]]
) -> Result[Option[T], E]:
    """Attempts to find the first element in the iterator that satisfies the given fallible predicate.

    Example:
    ```python
    ```
    """
