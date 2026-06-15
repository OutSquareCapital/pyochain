from collections.abc import Callable

import pytest

from pyochain import Option, Some


def test_map_basic_callable() -> None:
    assert Some(5).map(lambda x: x * 2).unwrap() == 10


def test_and_then_basic_callable() -> None:
    assert Some(5).and_then(lambda x: Some(x * 2)).unwrap() == 10


def test_pipe_basic_callable() -> None:
    assert Some(5).pipe(lambda opt: opt.unwrap() * 2) == 10


def test_args_no_kwargs() -> None:
    def add(a: int, b: int) -> int:  # noqa: FURB118
        return a + b

    assert Some(3).map(add, 5).unwrap() == 8
    assert Some(3).and_then(lambda x: Some(add(x, 5))).unwrap() == 8


def test_mixed_args_and_kwargs() -> None:
    def add(a: int, b: int, c: int = 7) -> int:
        return a + b + c

    assert Some(3).map(add, 5, c=11).unwrap() == 19
    assert Some(3).and_then(lambda x: Some(add(x, 5, c=11))).unwrap() == 19


def test_map_supports_high_arity_callable() -> None:
    def many_args(a: int, b: int, c: int, d: int, e: int, f: int) -> int:  # noqa: PLR0913, PLR0917
        return a + b + c + d + e + f

    assert Some(1).map(many_args, 2, 3, 4, 5, 6).unwrap() == 21


def test_pipe_supports_kwargs_on_self() -> None:
    def format_option(opt: Option[int], fmt: str = "decimal") -> str:
        return hex(opt.unwrap()) if fmt == "hex" else str(opt.unwrap())

    assert Some(255).pipe(format_option, fmt="hex") == "0xff"


def _raise_value_error(_x: int) -> int:
    msg = "map error"
    raise ValueError(msg)


def _raise_runtime_error(_x: int) -> Option[int]:
    msg = "and_then error"
    raise RuntimeError(msg)


def _raise_lookup_error(_x: Option[int]) -> int:
    msg = "into error"
    raise LookupError(msg)


@pytest.mark.parametrize(
    ("runner", "exception", "message"),
    [
        pytest.param(_raise_value_error, ValueError, "map error", id="map"),
        pytest.param(
            _raise_runtime_error, RuntimeError, "and_then error", id="and-then"
        ),
        pytest.param(_raise_lookup_error, LookupError, "into error", id="into"),
    ],
)
def test_exceptions_are_forwarded_unchanged(
    runner: Callable[[int], object], exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        _ = Some(5).map(runner)
