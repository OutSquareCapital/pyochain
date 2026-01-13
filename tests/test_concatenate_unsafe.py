"""Exhaustive test suite for concatenate() and concatenate_self() unsafe FFI functions.

This module tests all edge cases of the unsafe tuple building and function calling
in Rust FFI context for Option.map, Option.and_then, and Result.into methods.
"""

from collections.abc import Callable
from functools import partial

import pytest

import pyochain as pc


class TestConcatenateBasics:
    """Test basic concatenate functionality via Option/Result methods."""

    def test_concatenate_simple_function(self) -> None:
        """Test concatenate with simple pure function."""
        result = pc.Some(5).map(lambda x: x * 2)
        assert result.unwrap() == 10

    def test_concatenate_multiple_args(self) -> None:
        """Test concatenate with multiple positional arguments."""

        def add_three(a: int, b: int, c: int) -> int:
            return a + b + c

        result = pc.Some(5).map(add_three, 3, 2)
        assert result.unwrap() == 10

    def test_concatenate_with_kwargs(self) -> None:
        """Test concatenate with keyword arguments."""

        def func_with_kwargs(a: int, b: int = 10, c: int = 20) -> int:
            return a + b + c

        result = pc.Some(5).map(func_with_kwargs, b=15, c=25)
        assert result.unwrap() == 45

    def test_concatenate_mixed_args_kwargs(self) -> None:
        """Test concatenate with mixed positional and keyword arguments."""

        def func_mixed(a: int, b: int, c: int = 5) -> int:
            return a + b + c

        result = pc.Some(10).map(func_mixed, 20, c=30)
        assert result.unwrap() == 60


class TestConcatenateLambdas:
    """Test concatenate with various lambda expressions."""

    def test_simple_lambda(self) -> None:
        """Test concatenate with simple lambda."""
        result = pc.Some(5).map(lambda x: x + 10)
        assert result.unwrap() == 15

    def test_lambda_with_extra_args(self) -> None:
        """Test lambda with extra positional arguments."""
        result: pc.Option[int] = pc.Some(5).map(lambda x, y, z: x + y + z, 10, 20)  # type: ignore[arg-type]
        assert result.unwrap() == 35

    def test_lambda_with_kwargs(self) -> None:
        """Test lambda with keyword arguments."""
        result = pc.Some(5).map(lambda x, y=10: x + y, y=20)  # type: ignore[arg-type]
        assert result.unwrap() == 25

    def test_nested_lambdas(self) -> None:
        """Test nested lambda calls through chained map."""
        result = pc.Some(5).map(lambda x: lambda y: x + y).map(lambda f: f(10))  # type: ignore[arg-type]
        assert result.unwrap() == 15

    def test_lambda_with_captured_vars(self) -> None:
        """Test lambda that captures external variables."""
        multiplier = 5

        def closure_lambda(x: int) -> int:
            return x * multiplier

        result = pc.Some(10).map(closure_lambda)
        assert result.unwrap() == 50


class TestConcatenateClosures:
    """Test concatenate with closures and scope capture."""

    def test_simple_closure(self) -> None:
        """Test simple closure capturing outer variable."""

        def outer() -> Callable[..., int]:
            offset = 100

            def inner(x: int) -> int:
                return x + offset

            return inner

        func = outer()
        result = pc.Some(5).map(func)
        assert result.unwrap() == 105

    def test_closure_with_multiple_captures(self) -> None:
        """Test closure capturing multiple variables."""

        def outer() -> Callable[..., int]:
            a, b, c = 10, 20, 30

            def inner(x: int) -> int:
                return x + a + b + c

            return inner

        func = outer()
        result = pc.Some(5).map(func)
        assert result.unwrap() == 65

    def test_closure_modifying_capture(self) -> None:
        """Test closure that 'modifies' captured state (via return)."""

        def outer() -> Callable[..., int]:
            state = {"count": 0}

            def inner(x: int) -> int:
                state["count"] += 1
                return x * state["count"]

            return inner

        func = outer()
        result1 = pc.Some(5).map(func)
        result2 = pc.Some(10).map(func)
        assert result1.unwrap() == 5
        assert result2.unwrap() == 20

    def test_nested_closures(self) -> None:
        """Test deeply nested closures."""

        def level1() -> Callable[..., int]:
            a = 1

            def level2() -> Callable[..., int]:
                b = 2

                def level3(x: int) -> int:
                    return x + a + b

                return level3

            return level2()

        func = level1()
        result = pc.Some(10).map(func)
        assert result.unwrap() == 13

    def test_closure_with_class_context(self) -> None:
        """Test closure defined inside a class method."""

        class Calculator:
            def __init__(self, base: int) -> None:
                self.base = base

            def make_adder(self) -> Callable[..., int]:
                def adder(x: int) -> int:
                    return x + self.base

                return adder

        calc = Calculator(100)
        result = pc.Some(5).map(calc.make_adder())
        assert result.unwrap() == 105


class TestConcatenatePartials:
    """Test concatenate with functools.partial."""

    def test_simple_partial(self) -> None:
        """Test partial with one fixed argument."""

        def add(a: int, b: int) -> int:
            return a + b

        add_five = partial(add, 5)
        result = pc.Some(10).map(add_five)
        assert result.unwrap() == 15

    def test_partial_with_kwargs(self) -> None:
        """Test partial with fixed keyword arguments."""

        def func(a: int, b: int = 10, c: int = 20) -> int:
            return a + b + c

        partial_func = partial(func, b=30, c=40)
        result = pc.Some(5).map(partial_func)
        assert result.unwrap() == 75

    def test_partial_chained(self) -> None:
        """Test chaining multiple partials."""

        def three_arg_func(a: int, b: int, c: int) -> int:
            return a + b + c

        partial_one = partial(three_arg_func, 10)
        partial_two = partial(partial_one, 20)
        result = pc.Some(30).map(partial_two)
        assert result.unwrap() == 60

    def test_partial_with_extra_args(self) -> None:
        """Test partial that still accepts additional arguments."""

        def add_multiple(a: int, b: int, c: int) -> int:
            return a + b + c

        add_base = partial(add_multiple, 10)
        result = pc.Some(20).map(add_base, 30)
        assert result.unwrap() == 60


class TestConcatenateBuiltins:
    """Test concatenate with built-in functions."""

    def test_builtin_function(self) -> None:
        """Test with built-in functions."""
        result = pc.Some(5).map(str)
        assert result.unwrap() == "5"

    def test_builtin_with_args(self) -> None:
        """Test built-in function with extra arguments."""
        result = pc.Some(255).map(hex)
        assert result.unwrap() == "0xff"

    def test_builtin_method_reference(self) -> None:
        """Test built-in method references."""
        result = pc.Some("hello").map(str.upper)
        assert result.unwrap() == "HELLO"


class TestConcatenateCallables:
    """Test concatenate with various callable types."""

    def test_class_as_callable(self) -> None:
        """Test class constructor as callable."""

        class Container:
            def __init__(self, value: int) -> None:
                self.value = value

        result = pc.Some(42).map(Container)
        obj = result.unwrap()
        assert obj.value == 42

    def test_callable_object(self) -> None:
        """Test object with __call__ method."""

        class Multiplier:
            def __init__(self, factor: int) -> None:
                self.factor = factor

            def __call__(self, x: int) -> int:
                return x * self.factor

        multiplier = Multiplier(5)
        result = pc.Some(10).map(multiplier)
        assert result.unwrap() == 50

    def test_staticmethod(self) -> None:
        """Test staticmethod as callable."""

        class MyClass:
            @staticmethod
            def double(x: int) -> int:
                return x * 2

        result = pc.Some(21).map(MyClass.double)
        assert result.unwrap() == 42

    def test_classmethod(self) -> None:
        """Test classmethod as callable."""

        class MyClass:
            multiplier = 3

            @classmethod
            def multiply(cls, x: int) -> int:
                return x * cls.multiplier

        result = pc.Some(7).map(MyClass.multiply)
        assert result.unwrap() == 21

    def test_method_reference(self) -> None:
        """Test bound method as callable."""

        class Processor:
            def __init__(self, base: int) -> None:
                self.base = base

            def process(self, x: int) -> int:
                return x + self.base

        proc = Processor(100)
        result = pc.Some(5).map(proc.process)
        assert result.unwrap() == 105


class TestConcatenateAndThen:
    """Test concatenate_self behavior via and_then."""

    def test_and_then_simple(self) -> None:
        """Test and_then with simple function."""

        def make_option(x: int) -> pc.Option[int]:
            return pc.Some(x * 2)

        result = pc.Some(5).and_then(make_option)
        assert result.unwrap() == 10

    def test_and_then_with_args(self) -> None:
        """Test and_then with extra arguments."""

        def combine(a: int, b: int) -> pc.Option[int]:
            return pc.Some(a + b)

        result = pc.Some(5).and_then(combine, 10)
        assert result.unwrap() == 15

    def test_and_then_with_kwargs(self) -> None:
        """Test and_then with keyword arguments."""

        def func_kwargs(x: int, multiplier: int = 2) -> pc.Option[int]:
            return pc.Some(x * multiplier)

        result = pc.Some(5).and_then(func_kwargs, multiplier=3)
        assert result.unwrap() == 15

    def test_and_then_returns_none(self) -> None:
        """Test and_then that returns NONE."""

        def maybe_double(x: int) -> pc.Option[int]:
            if x > 100:
                return pc.NONE
            return pc.Some(x * 2)

        result1 = pc.Some(50).and_then(maybe_double)
        result2 = pc.Some(150).and_then(maybe_double)

        assert result1.unwrap() == 100
        assert result2.is_none()

    def test_and_then_chain(self) -> None:
        """Test chaining multiple and_then calls."""

        def add_one(x: int) -> pc.Option[int]:
            return pc.Some(x + 1)

        def double(x: int) -> pc.Option[int]:
            return pc.Some(x * 2)

        result = pc.Some(5).and_then(add_one).and_then(double)
        assert result.unwrap() == 12


class TestConcatenateInto:
    """Test concatenate_self behavior via into()."""

    def test_into_with_function(self) -> None:
        """Test into with simple function - receives full Some object."""

        def extract_and_triple(opt: pc.Option[int]) -> list[int]:
            val = opt.unwrap()
            return [val, val * 2, val * 3]

        result = pc.Some(5).into(extract_and_triple)
        assert result == [5, 10, 15]

    def test_into_with_kwargs(self) -> None:
        """Test into with keyword arguments - receives full Some object."""

        def format_option(opt: pc.Option[int], fmt: str = "decimal") -> str:
            x = opt.unwrap()
            if fmt == "hex":
                return hex(x)
            return str(x)

        result1 = pc.Some(255).into(format_option)
        result2 = pc.Some(255).into(format_option, fmt="hex")

        assert result1 == "255"
        assert result2 == "0xff"

    def test_into_with_builtin(self) -> None:
        """Test into with identity function."""
        result = pc.Some(5).into(lambda x: x)
        # into() receives the full Some object
        assert isinstance(result, pc.Option)

    def test_into_with_external_function(self) -> None:
        """Test into calling external API - receives full Some object."""

        def external_api(opt: pc.Option[int]) -> dict[str, int]:
            x = opt.unwrap()
            return {"input": x, "output": x * 2}

        result = pc.Some(10).into(external_api)
        assert result == {"input": 10, "output": 20}

    def test_into_result_ok(self) -> None:
        """Test into with Result.Ok - receives full Ok object."""

        def process(res: pc.Result[int, object]) -> str:
            x = res.unwrap()
            return f"Value: {x}"

        result = pc.Ok(42).into(process)
        assert result == "Value: 42"

    def test_into_with_lambda(self) -> None:
        """Test into with lambda - receives full Some object."""
        result = pc.Some(5).into(lambda opt: opt.unwrap() * 10)
        assert result == 50

    def test_into_closure(self) -> None:
        """Test into with closure - receives full Some object."""

        def make_formatter(prefix: str) -> Callable[..., str]:
            def formatter(opt: pc.Option[int]) -> str:
                return f"{prefix}: {opt.unwrap()}"

            return formatter

        formatter = make_formatter("Number")
        result = pc.Some(42).into(formatter)
        assert result == "Number: 42"


class TestConcatenateErrorHandling:
    """Test error handling in concatenate functions."""

    def test_function_raises_exception(self) -> None:
        """Test when mapped function raises exception."""

        def failing_func(_x: int) -> int:
            msg = "Test error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Test error"):
            pc.Some(5).map(failing_func).unwrap()

    def test_function_with_type_error(self) -> None:
        """Test when function call causes type error."""

        def type_strict(x: int) -> int:
            return x + "string"  # type: ignore[return-value]

        try:
            pc.Some(5).map(type_strict).unwrap()
            msg = "Should have raised TypeError"
            raise AssertionError(msg)
        except TypeError:
            pass

    def test_and_then_with_exception(self) -> None:
        """Test and_then when function raises."""

        def failing_func(_x: int) -> pc.Option[int]:
            msg = "Test error"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="Test error"):
            pc.Some(5).and_then(failing_func)

    def test_into_with_exception(self) -> None:
        """Test into when function raises."""

        def failing_func(_x: pc.Some[int]) -> int:
            msg = "Into error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Into error"):
            pc.Some(5).into(failing_func)


class TestConcatenateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_args(self) -> None:
        """Test with no extra arguments."""

        def zero_arg_wrapper(x: int) -> int:
            return x

        result = pc.Some(42).map(zero_arg_wrapper)
        assert result.unwrap() == 42

    def test_many_args(self) -> None:
        """Test with many arguments."""

        def many_args(a: int, b: int, c: int, d: int, e: int, f: int) -> int:  # noqa: PLR0913
            return a + b + c + d + e + f

        result = pc.Some(1).map(many_args, 2, 3, 4, 5, 6)
        assert result.unwrap() == 21

    def test_none_value(self) -> None:
        """Test mapping over None values."""

        def func(x: int | None) -> int:
            if x is None:
                return 0
            return x

        result = pc.Some(None).map(func)
        assert result.unwrap() == 0

    def test_complex_nested_structures(self) -> None:
        """Test with complex nested data structures."""

        def process_nested(data: dict[str, list[int]]) -> int:
            return sum(data["values"])

        test_data = {"values": [1, 2, 3, 4, 5]}
        result = pc.Some(test_data).map(process_nested)
        assert result.unwrap() == 15

    def test_recursive_function(self) -> None:
        """Test with recursive function."""

        def factorial(n: int) -> int:
            if n <= 1:
                return 1
            return n * factorial(n - 1)

        result = pc.Some(5).map(factorial)
        assert result.unwrap() == 120

    def test_generator_function(self) -> None:
        """Test with generator function wrapped."""

        def gen_func(x: int) -> list[int]:
            return [i * x for i in range(5)]

        result = pc.Some(3).map(gen_func)
        assert result.unwrap() == [0, 3, 6, 9, 12]

    def test_none_returned_as_value(self) -> None:
        """Test function that returns None as a value."""

        def return_none(_x: int) -> None:
            return None

        result = pc.Some(5).map(return_none)
        assert result.unwrap() is None

    def test_exception_in_kwargs_evaluation(self) -> None:
        """Test exception during kwargs evaluation."""

        def might_fail(x: int, value: int | None = None) -> int:
            return x + (value or 0)

        result = pc.Some(5).map(might_fail, value=10)
        assert result.unwrap() == 15


class TestConcatenateWithResult:
    """Test concatenate behavior with Result types."""

    def test_result_ok_map(self) -> None:
        """Test map on Result.Ok."""

        def double(x: int) -> int:
            return x * 2

        result = pc.Ok(5).map(double)
        assert result.unwrap() == 10

    def test_result_err_map_noop(self) -> None:
        """Test that map is noop on Result.Err."""

        def double(x: int) -> int:
            return x * 2

        result = pc.Err("error message").map(double)
        error = result.unwrap_err()
        assert error == "error message"

    def test_result_ok_and_then(self) -> None:
        """Test and_then on Result.Ok."""

        def safe_divide(x: int) -> pc.Result[int, str]:
            if x == 0:
                return pc.Err("Division by zero")
            return pc.Ok(100 // x)

        result1 = pc.Ok(10).and_then(safe_divide)
        result2 = pc.Ok(0).and_then(safe_divide)

        assert result1.unwrap() == 10
        assert result2.unwrap_err() == "Division by zero"

    def test_result_ok_into(self) -> None:
        """Test into on Result.Ok - receives full Ok object."""

        def format_result(res: pc.Result[int, object]) -> str:
            x = res.unwrap()
            return f"Result: {x}"

        result = pc.Ok(42).into(format_result)
        assert result == "Result: 42"


class TestConcatenatePerformance:
    """Test performance characteristics and regression checks."""

    def test_many_iterations(self) -> None:
        """Test stability through many iterations."""

        def increment(x: int) -> int:
            return x + 1

        result = pc.Some(0)
        for _ in range(100):
            result = result.map(increment)

        assert result.unwrap() == 100

    def test_large_argument_list(self) -> None:
        """Test with large argument lists."""

        def sum_all(*args: int) -> int:
            return sum(args)

        args = list(range(50))
        result = pc.Some(args[0]).map(sum_all, *args[1:])
        assert result.unwrap() == sum(range(50))
