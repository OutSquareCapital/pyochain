from __future__ import annotations

from typing import Any, assert_never, assert_type

from pyochain import Err, Null, Ok, Option, Result, Some, option

from ._utils import Animal, AnimalLit, Dog, LitCat, LitDog


def check_covariance() -> None:
    opt: Option[Animal] = Some(Dog())
    _ = assert_type(opt, Option[Dog])


def check_option_basic() -> None:
    base = assert_type(Some(Dog()), Option[Dog])
    canary = assert_type(base.unwrap_or_none(), Dog | None)
    _ = assert_type(base.map(Animal.as_parent), Option[Animal])
    if canary is not None:
        _ = assert_type(canary, Dog)


def check_option_transpose() -> None:
    _a: Result[Option[int], int] = Some(Ok(10)).transpose()
    _b: Result[Option[int], int] = Some(Err(10)).transpose()
    _c: Result[Option[int], int] = Null().transpose()


def check_option_literal() -> None:
    lit = assert_type(_get_cat(), AnimalLit | None)
    # Inferred as Option[str]
    _ = assert_type(option(lit), Option[str])
    # Need to add explicit type hint to get Option[AnimalLit]
    opt_casted: Option[AnimalLit] = assert_type(option(lit), Option[AnimalLit])
    _ = assert_type(opt_casted.map(_literal), Option[AnimalLit])
    # Issue: Literals aren't handled for type unions, even if both members are covariant.
    # pyrefly: ignore [non-exhaustive-match]
    match opt_casted:  # pyright: ignore[reportMatchNotExhaustive]
        case Some("dog") as opt_casted:
            # pyrefly: ignore [assert-type]
            _ = assert_type(opt_casted.unwrap(), LitDog)  # pyright: ignore[reportAssertTypeFailure]
        case Some("cat"):
            # pyrefly: ignore[assert-type]
            _ = assert_type(opt_casted.unwrap(), LitCat)  # pyright: ignore[reportAssertTypeFailure]
        case Some("tyrannosaurus"):  # pyright: ignore[reportUnnecessaryComparison]
            _ = assert_never(opt_casted.unwrap())  # pyright: ignore[reportUnreachable]
        case Null():
            _ = assert_type(opt_casted, Null[AnimalLit])


def check_option_flatten() -> None:
    def _(x: int) -> int:
        return x

    _a = assert_type(Some(Some(10)).flatten(), Option[int])
    _b = assert_type(Some(Null()).flatten(), Option[Any])
    _c = assert_type(Null().flatten(), Option[Any])
    _d = assert_type(Some(Null()).flatten().map(_), Option[int])
    _e = assert_type(Some(Some(Some(10))).flatten().flatten().map(str), Option[str])


def check_option_and_then() -> None:
    """Rust equivalent who compiles (the type hints for variables have been added *last*, so they are not helping for inference):

    ```rust
    let _a: Option<i32> = Some(10).and_then(Some);
    let _b: Option<Option<i32>> = Some::<Option<i32>>(None).and_then(Some);
    let _c: Option<i32> = None::<i32>.and_then(Some);
    ```
    """
    _a = assert_type(Some(10).and_then(Some), Option[int])
    _b = assert_type(Some[Option[int]](Null()).and_then(Some), Option[Option[int]])
    _c = assert_type(Null[int]().and_then(Some), Option[int])


def _get_cat() -> AnimalLit | None:
    return "cat"


def _literal(x: AnimalLit) -> AnimalLit:
    return x
