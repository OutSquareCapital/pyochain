import pytest

from pyochain import Err, Ok, Result, ResultUnwrapError


def test_ok_map() -> None:
    assert Ok(5).map(lambda x: x * 2).unwrap() == 10


def test_err_map_noop() -> None:
    assert Err("error message").map(lambda x: x).unwrap_err() == "error message"  # pyright: ignore[reportAny]


def test_ok_and_then() -> None:

    def safe_divide(x: int) -> Result[int, str]:
        if x == 0:
            return Err("Division by zero")
        return Ok(100 // x)

    assert Ok(10).and_then(safe_divide).unwrap() == 10
    assert Ok(0).and_then(safe_divide).unwrap_err() == "Division by zero"


def test_ok_into() -> None:
    assert Ok(42).into(lambda res: f"Result: {res.unwrap()}") == "Result: 42"


def test_err_unwrap_formats_python_exception_readably() -> None:

    with pytest.raises(ResultUnwrapError) as exc_info:
        Err(ValueError("line 1\nline 2")).unwrap()

    assert (
        str(exc_info.value) == "called `unwrap` on an `Err`: ValueError: line 1\nline 2"
    )


def test_err_expect_formats_python_exception_readably() -> None:

    with pytest.raises(ResultUnwrapError) as exc_info:
        Err(ValueError("line 1\nline 2")).expect("custom context")

    assert str(exc_info.value) == "custom context: ValueError: line 1\nline 2"


def test_err_unwrap_keeps_repr_for_non_exception_values() -> None:

    with pytest.raises(ResultUnwrapError) as exc_info:
        Err({"code": 12}).unwrap()

    assert str(exc_info.value) == "called `unwrap` on an `Err`: {'code': 12}"
