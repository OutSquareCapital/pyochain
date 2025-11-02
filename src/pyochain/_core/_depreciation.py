import warnings
from collections.abc import Callable
from functools import wraps


def deprecated[**P, R](msg: str):
    def decorator(func: Callable[P, R]):
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
