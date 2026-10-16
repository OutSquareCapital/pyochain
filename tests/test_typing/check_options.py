from __future__ import annotations

from typing import Any, Literal, assert_never, assert_type

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


def check_option_lit_matchs() -> None:
    """The match cases do work as expected.

    However (more a *wish* than a *need*), the original `Option` type isn't inferred after the match.

    This also extends to containers.\\
    So even tough we know that `Some.value` **IS** a `LitDog`, `Some.unwrap()` stays as `AnimalLit`.
    """
    opt_casted = check_opt_lit_inference()
    _ = assert_type(opt_casted.map(_literal), Option[AnimalLit])
    # pyrefly: ignore [non-exhaustive-match]
    match opt_casted:  # pyright: ignore[reportMatchNotExhaustive]
        case Some("dog" as dog):
            _ = assert_type(dog, LitDog)
            # pyrefly: ignore [assert-type]
            _ = assert_type(opt_casted.unwrap(), LitDog)  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore[type-assertion-failure]
        case Some("cat" as cat):
            _ = assert_type(cat, LitCat)
            # pyrefly: ignore [assert-type]
            _ = assert_type(opt_casted.unwrap(), LitCat)  # pyright: ignore[reportAssertTypeFailure]  # ty: ignore[type-assertion-failure]
        case Some("tyrannosaurus" as trex):  # pyright: ignore[reportUnnecessaryComparison, reportUnknownVariableType]
            _ = assert_never(trex)
            _ = assert_never(opt_casted.unwrap())  # pyright: ignore[reportUnreachable]
        case Null():
            _ = assert_type(opt_casted, Null[AnimalLit])
            _ = assert_never(opt_casted.unwrap())


def check_opt_lit_inference() -> Option[AnimalLit]:
    lit = assert_type(_get_cat(), AnimalLit | None)
    # NOTE: specific to basedpyright
    # Inferred as Option[str]
    # pyrefly: ignore [assert-type]
    _ = assert_type(option(lit), Option[str])  # ty: ignore[type-assertion-failure]
    # Need to add explicit type hint to get Option[AnimalLit]
    opt_casted: Option[AnimalLit] = assert_type(option(lit), Option[AnimalLit])
    # ty does better and infer directly the output as Option[AnimalLit]
    _ = assert_type(option(lit), Option[AnimalLit])  # pyright: ignore[reportAssertTypeFailure]
    return opt_casted


def check_narrowed_overloads() -> None:
    opt = Some(Dog())
    _ = assert_type(opt.unwrap_or_none(), Dog | None)
    _ = assert_type(opt.unwrap_or(str(1)), Dog | str)
    if isinstance(opt, Some):
        _ = assert_type(opt.unwrap(), Dog)
        _ = assert_type(opt.is_some(), Literal[True])
        _ = assert_type(opt.is_none(), Literal[False])
        _ = assert_type(opt.unwrap_or_none(), Dog)
        _ = assert_type(opt.unwrap_or(str(1)), Dog)
        _ = assert_type(opt.map(Animal.as_parent), Some[Animal])
        _a: Ok[Dog, str] = opt.ok_or("err")
        _ = assert_type(opt.unwrap_or_else(Animal), Dog)
        _ = assert_type(opt.and_(Some("x")), Option[str])
        _ = assert_type(opt.or_(Some("x")), Some[Dog])
        _ = assert_type(opt.or_else(lambda: Some("x")), Some[Dog])
        _b: Ok[Dog, str] = opt.ok_or_else(lambda: "err")
        _ = assert_type(opt.inspect(print), Some[Dog])
    else:
        _ = assert_type(opt.is_some(), Literal[False])
        _ = assert_type(opt.is_none(), Literal[True])
        _ = assert_type(opt.unwrap_or_none(), None)
        _ = assert_type(opt.unwrap_or(str(1)), str)
        _ = assert_type(opt.map(Animal.as_parent), Null[Animal])
        _ = assert_type(opt.ok_or("err"), Err[Dog, str])
        _ = assert_type(opt.unwrap_or_else(Animal), Animal)
        _ = assert_type(opt.and_(Some("x")), Null[str])
        _ = assert_type(opt.or_(Some("x")), Option[str])
        _ = assert_type(opt.or_else(lambda: Some("x")), Option[str])
        _ = assert_type(opt.ok_or_else(lambda: "err"), Err[Dog, str])
        _ = assert_type(opt.inspect(print), Null[Dog])
        _ = assert_never(opt.unwrap())


def check_option_flatten() -> None:
    def _(x: int) -> int:
        return x

    _a = assert_type(Some(Some(10)).flatten(), Option[int])
    _b = assert_type(Some(Null()).flatten(), Option[Any])
    _c = assert_type(Null().flatten(), Option[Any])
    _d = assert_type(Some(Null()).flatten().map(_), Option[int])
    _e = assert_type(Some(Some(Some(10))).flatten().flatten().map(str), Option[str])


def check_option_unzip() -> None:
    opt = Some((Dog(), Animal()))
    if isinstance(opt, Some):
        _ = assert_type(opt.unzip(), tuple[Some[Dog], Some[Animal]])
    else:
        _ = assert_type(opt.unzip(), tuple[Null[Dog], Null[Animal]])


def check_option_zip_family() -> None:
    opt_dog: Option[Dog] = Some(Dog())
    opt_animal: Option[Animal] = Some(Animal())
    if isinstance(opt_dog, Some) and isinstance(opt_animal, Some):
        _ = assert_type(opt_dog.zip(opt_animal), Some[tuple[Dog, Animal]])
        _ = assert_type(opt_dog.zip_with(opt_animal, lambda _, a: a), Some[Animal])
    _ = assert_type(opt_dog.zip(opt_animal), Option[tuple[Dog, Animal]])
    _ = assert_type(opt_dog.zip_with(opt_animal, lambda _, a: a), Option[Animal])


def check_option_reduce() -> None:
    def pick_second(_: Dog, b: Animal) -> Animal:
        return b

    opt_dog: Option[Dog] = Some(Dog())
    opt_animal: Option[Animal] = Some(Animal())
    if isinstance(opt_dog, Some) and isinstance(opt_animal, Some):
        _ = assert_type(opt_dog.reduce(opt_animal, pick_second), Some[Animal])
    if isinstance(opt_dog, Some) and isinstance(opt_animal, Null):
        _ = assert_type(opt_dog.reduce(opt_animal, pick_second), Some[Dog])
    if isinstance(opt_dog, Null) and isinstance(opt_animal, Some):
        _ = assert_type(opt_dog.reduce(opt_animal, pick_second), Some[Animal])
    if isinstance(opt_dog, Null) and isinstance(opt_animal, Null):
        _ = assert_type(opt_dog.reduce(opt_animal, pick_second), Null[Animal])


def check_option_xor() -> None:
    opt_a: Option[Dog] = Some(Dog())
    opt_b: Option[Dog] = Null()
    if isinstance(opt_a, Some) and isinstance(opt_b, Null):
        _ = assert_type(opt_a.xor(opt_b), Some[Dog])
    if isinstance(opt_a, Null) and isinstance(opt_b, Some):
        _ = assert_type(opt_a.xor(opt_b), Some[Dog])
    if isinstance(opt_a, Some) and isinstance(opt_b, Some):
        _ = assert_type(opt_a.xor(opt_b), Null[Dog])
    if isinstance(opt_a, Null) and isinstance(opt_b, Null):
        _ = assert_type(opt_a.xor(opt_b), Null[Dog])


def check_option_and_then() -> None:
    """Rust equivalent who compiles (the type hints for variables have been added *last*, so they are not helping for inference):

    ```rust
    let _a: Option<i32> = Some(10).and_then(Some);
    let _b: Option<Option<i32>> = Some::<Option<i32>>(None).and_then(Some);
    let _c: Option<i32> = None::<i32>.and_then(Some);
    ```
    """
    _a = assert_type(Some(10).and_then(Some), Option[int])  # ty: ignore[type-assertion-failure]
    # ty infer the Literal
    _a_ty = assert_type(Some(10).and_then(Some), Option[Literal[10]])  # pyright: ignore[reportAssertTypeFailure]
    _b = assert_type(Some[Option[int]](Null()).and_then(Some), Option[Option[int]])
    _c = assert_type(Null[int]().and_then(Some), Option[int])


def _get_cat() -> AnimalLit | None:
    return "cat"


def _literal(x: AnimalLit) -> AnimalLit:
    return x
