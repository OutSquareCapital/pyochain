from __future__ import annotations

from typing import Any, assert_type

from pyochain import NONE, Err, Null, Ok, Option, Result, Some

from ._utils import Animal, Dog, identity


def check_covariance() -> None:
    res: Result[Animal, Any] = Ok(Dog()).map(lambda x: x)
    _ = assert_type(res, Result[Dog, Any])


def check_result_basic() -> None:
    ok = assert_type(Ok(Dog()), Result[Dog, Any])
    err = assert_type(Err(Dog()), Result[Any, Dog])
    _ = assert_type(Ok[int, str](42), Result[int, str])
    _a = assert_type(ok.map(identity).map_err(Animal.as_parent), Result[Dog, Animal])
    _b = assert_type(ok.map_err(identity).map(Animal.as_parent), Result[Animal, Any])
    _c = assert_type(err.map(identity).map_err(Animal.as_parent), Result[Any, Animal])
    _d = assert_type(err.map_err(identity).map(Animal.as_parent), Result[Animal, Dog])


def check_result_transpose() -> None:
    a = assert_type(Ok(Some(10)), Result[Option[int], Any])
    _a = assert_type(a.transpose(), Option[Result[int, Any]])
    b = assert_type(Err(Some(10)), Result[Any, Option[int]])
    _b = assert_type(b.transpose(), Option[Result[Any, Option[int]]])
    c = assert_type(Ok[Option[int], int](NONE), Result[Option[int], int])
    _c = assert_type(c.transpose(), Option[Result[int, int]])
    d = Err[Option[int], Option[int]](Null())
    _ = assert_type(d, Result[Option[int], Option[int]])
    _d = assert_type(d.transpose(), Option[Result[int, Option[int]]])


def check_result_flatten() -> None:
    """Rust equivalent who compiles (the type hints for variables have been added *last*, so they are not helping for inference):

    ```rust

    let _a: Result<i32, i32> = Ok(Ok::<i32, i32>(10)).flatten();
    let _b: Result<&str, &str> = Ok(Err::<&str, &str>("error")).flatten();
    ```
    """
    _a = assert_type(Ok(Ok[int, int](10)).flatten(), Result[int, int])
    _b = assert_type(Ok(Err[str, str]("error")).flatten(), Result[str, str])
    _ = assert_type(Err(Err("error")), Result[Any, Result[Any, str]])


def check_and_then_result() -> None:
    """Rust equivalent who compiles (the type hints for variables have been added *last*, so they are not helping for inference):

    ```rust
    fn test_flatten() {
    let _a: Result<i32, i32> = Ok(Ok::<i32, i32>(10)).and_then(|x| x);
    let _b: Result<&str, &str> = Ok(Err::<&str, &str>("error")).and_then(|x| x);
    let _c: Result<&str, &str> = Err::<&str, &str>("error").and_then(|x| Ok(x));
    let _d: Result<Result<&str, &str>, Result<&str, &str>> =
        Err(Err::<&str, &str>("error")).and_then(|x: Result<&str, &str>| Ok(x));
    let _e: Result<Result<i32, i32>, Result<i32, i32>> =
        Err(Ok::<i32, i32>(10)).and_then(|x: Result<i32, i32>| Ok(x));
    }
    ```
    """
    msg = "error"
    a = Ok[Result[int, int], int](Ok[int, int](10)).and_then(lambda x: x)
    b = Ok[Result[str, str], str](Err[str, str](msg)).and_then(lambda x: x)
    c = Err[str, str](msg).and_then(Ok)
    d = Err(Err[str, str](msg)).and_then(_fn_str)
    e = Err(Ok[int, int](10)).and_then(_fn_int)
    _ = assert_type(a, Result[int, int])
    _ = assert_type(b, Result[str, str])
    _ = assert_type(c, Result[str, str])
    _ = assert_type(d, Result[Result[str, str], Result[str, str]])
    _ = assert_type(e, Result[Result[int, int], Result[int, int]])


def _fn_str(x: Result[str, str]) -> Result[Result[str, str], Any]:
    return Ok(x)


def _fn_int(x: Result[int, int]) -> Result[Result[int, int], Any]:
    return Ok(x)
