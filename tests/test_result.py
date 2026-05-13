import pytest

from pyochain import Err, Ok, Result, ResultUnwrapError


def test_ok_map() -> None:
    """Test map on Result.Ok."""

    def double(x: int) -> int:
        return x * 2

    result = Ok(5).map(double)
    assert result.unwrap() == 10


def test_err_map_noop() -> None:
    """Test that map is noop on Result.Err."""

    def double(x: int) -> int:
        return x * 2

    result = Err("error message").map(double)
    error = result.unwrap_err()
    assert error == "error message"


def test_ok_and_then() -> None:
    """Test and_then on Result.Ok."""

    def safe_divide(x: int) -> Result[int, str]:
        if x == 0:
            return Err("Division by zero")
        return Ok(100 // x)

    result1 = Ok(10).and_then(safe_divide)
    result2 = Ok(0).and_then(safe_divide)

    assert result1.unwrap() == 10
    assert result2.unwrap_err() == "Division by zero"


def test_ok_into() -> None:
    """Test into on Result.Ok - receives full Ok object."""

    def format_result(res: Result[int, object]) -> str:
        x = res.unwrap()
        return f"Result: {x}"

    result = Ok(42).into(format_result)
    assert result == "Result: 42"


def test_err_unwrap_formats_python_exception_readably() -> None:
    """Test unwrap formatting for Python exception payloads."""
    result: Result[int, Exception] = Err(ValueError("line 1\nline 2"))

    with pytest.raises(ResultUnwrapError) as exc_info:
        result.unwrap()

    assert (
        str(exc_info.value) == "called `unwrap` on an `Err`: ValueError: line 1\nline 2"
    )


def test_err_expect_formats_python_exception_readably() -> None:
    """Test expect formatting for Python exception payloads."""
    result: Result[int, Exception] = Err(ValueError("line 1\nline 2"))

    with pytest.raises(ResultUnwrapError) as exc_info:
        result.expect("custom context")

    assert str(exc_info.value) == "custom context: ValueError: line 1\nline 2"


def test_err_unwrap_keeps_repr_for_non_exception_values() -> None:
    """Test unwrap formatting for non-exception error payloads."""
    result: Result[int, object] = Err({"code": 12})

    with pytest.raises(ResultUnwrapError) as exc_info:
        result.unwrap()

    assert str(exc_info.value) == "called `unwrap` on an `Err`: {'code': 12}"
