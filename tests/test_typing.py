from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Never, assert_type

from pyochain import (
    NONE,
    Dict,
    Err,
    Null,
    Ok,
    Option,
    Range,
    Result,
    Seq,
    Set,
    Some,
    Vec,
    option,
)
from pyochain.abc import PyoIterator

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Container,
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Mapping,
        MappingView,
        MutableMapping,
        MutableSequence,
        Reversible,
        Sized,
        ValuesView,
    )

    from pyochain import Peekable
    from pyochain.abc import (
        PyoCollection,
        PyoContainer,
        PyoItemsView,
        PyoIterable,
        PyoKeysView,
        PyoMapping,
        PyoMappingView,
        PyoMutableMapping,
        PyoMutableSequence,
        PyoReversible,
        PyoSequence,
        PyoSet,
        PyoSized,
        PyoValuesView,
    )


@dataclass
class Animal:
    pass


@dataclass
class Dog(Animal):
    pass


type LitDog = Literal["dog"]
type LitCat = Literal["cat"]
type AnimalLit = Literal["dog", "cat"]


def check_iterables_covariance() -> None:
    base = Vec[Dog]()
    _abc_iterable: PyoIterable[Animal] = _foo(base)
    _abc_iterator: PyoIterator[Animal] = _foo(base.iter())
    _abc_collection: PyoCollection[Animal] = _foo(base)
    _abc_sequence: PyoSequence[Animal] = _foo(base)
    _peekable_iterator: Peekable[Animal] = base.iter().peekable()
    _abc_set_immutable: PyoSet[Animal] = _foo(base.pipe(Set))
    _seq_immutable: Seq[Animal] = base.pipe(Seq)
    # pyrefly: ignore [bad-assignment]
    _: PyoMutableSequence[Animal] = _foo(base)  # pyright: ignore[reportAssignmentType]


def _foo[T: Iterable[Animal]](x: T) -> T:
    return x


def check_monads_covariance() -> None:
    opt = assert_type(Some(Dog()), Option[Dog])
    res = assert_type(Ok(Dog()), Result[Dog, Any])
    _as_opt: Option[Animal] = assert_type(opt, Option[Dog])
    _as_res: Result[Animal, str] = assert_type(res.map(lambda x: x), Result[Dog, Any])


def check_option_basic() -> None:
    base = assert_type(Some(Dog()), Option[Dog])
    canary = assert_type(base.unwrap_or_none(), Dog | None)
    _ = assert_type(base.map(_value), Option[Animal])
    if canary is not None:
        _ = assert_type(canary, Dog)


def _value(x: Animal) -> Animal:
    return x


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
            _ = assert_type(opt_casted.unwrap(), Never)  # pyright: ignore[reportUnreachable]
        case Null():
            _ = assert_type(opt_casted, Null[AnimalLit])


def _get_cat() -> AnimalLit | None:
    return "cat"


def _literal(x: AnimalLit) -> AnimalLit:
    return x


def check_result_basic() -> None:
    ok = assert_type(Ok(Dog()), Result[Dog, Any])
    err = assert_type(Err(Dog()), Result[Any, Dog])
    _ = assert_type(Ok[int, str](42), Result[int, str])
    _a = assert_type(ok.map(_identity).map_err(_value), Result[Dog, Animal])
    _b = assert_type(ok.map_err(_identity).map(_value), Result[Animal, Any])
    _c = assert_type(err.map(_identity).map_err(_value), Result[Any, Animal])
    _d = assert_type(err.map_err(_identity).map(_value), Result[Animal, Dog])


def check_result_transpose() -> None:
    a = assert_type(Ok(Some(10)), Result[Option[int], Any])
    _a = assert_type(a.transpose(), Option[Result[int, Any]])
    b = assert_type(Err(Some(10)), Result[Any, Option[int]])
    _b = assert_type(b.transpose(), Option[Result[Any, Option[int]]])
    c = assert_type(Ok[Option[int], int](NONE), Result[Option[int], int])
    _c = assert_type(c.transpose(), Option[Result[int, int]])
    d = assert_type(
        Err[Option[int], Option[int]](Null()), Result[Option[int], Option[int]]
    )
    _d = assert_type(d.transpose(), Option[Result[int, Option[int]]])


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
    _a = assert_type(
        Ok[Result[int, int], int](Ok[int, int](10)).and_then(lambda x: x),
        Result[int, int],
    )
    _b = assert_type(
        Ok[Result[str, str], str](Err[str, str](msg)).and_then(lambda x: x),
        Result[str, str],
    )
    _c = assert_type(Err[str, str](msg).and_then(Ok), Result[str, str])
    _d = assert_type(
        Err(Err[str, str](msg)).and_then(_fn_str),
        Result[Result[str, str], Result[str, str]],
    )
    _e = assert_type(
        Err(Ok[int, int](10)).and_then(_fn_int),
        Result[Result[int, int], Result[int, int]],
    )


def _fn_str(x: Result[str, str]) -> Result[Result[str, str], Any]:
    return Ok(x)


def _fn_int(x: Result[int, int]) -> Result[Result[int, int], Any]:
    return Ok(x)


def check_iter_flatten() -> None:
    nested = (
        Range(3)
        .iter()
        .map(
            lambda x: (
                Range(x)
                .iter()
                .map(lambda y: Range(y).iter().map(lambda z: Range(z).pipe(list)))
            )
        )
    )
    _ = assert_type(nested, PyoIterator[PyoIterator[PyoIterator[list[int]]]])
    one = assert_type(nested.flatten(), PyoIterator[PyoIterator[list[int]]])
    two = assert_type(one.flatten(), PyoIterator[list[int]])
    ok = assert_type(two.flatten(), PyoIterator[int])
    # Expected to fail
    _fail = ok.flatten()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]


def check_iterable_args(base: PyoIterable[Dog], canary: Iterable[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_iterator_args(base: PyoIterator[Dog], canary: Iterator[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)
    _ = _iterator(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_sized_args(base: PyoSized, canary: Sized) -> None:
    # pyrefly: ignore [bad-argument-type]
    _ = _iterable(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterable(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_reversible_args(base: PyoReversible[Dog], canary: Reversible[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    _ = _reversible(base)
    _ = _reversible(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_container_args(base: PyoContainer[Animal], canary: Container[Animal]) -> None:
    # pyrefly: ignore [bad-argument-type]
    _ = _iterable(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterable(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    _ = _container(base)
    _ = _container(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_collection_args(base: PyoCollection[Dog], canary: Collection[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)
    _ = _collection(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_sequence_args(base: PyoSequence[Dog], canary: Sequence[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)
    _ = _sequence(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_mutable_sequence_args(
    base: PyoMutableSequence[Dog], canary: MutableSequence[Dog]
) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)
    _ = _sequence(canary)
    _ = _mutable_sequence(base)
    _ = _mutable_sequence(canary)


def check_mapping(
    base: PyoMapping[Animal, Animal], canary: Mapping[Animal, Animal]
) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)
    _ = _collection(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mapping(base)
    _ = _mapping(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_mapping(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_mapping(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mapping_view(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mapping_view(canary)  # pyright: ignore[reportArgumentType]


def check_mutable_mapping(
    base: PyoMutableMapping[Animal, Animal], canary: MutableMapping[Animal, Animal]
) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)
    _ = _collection(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mapping(base)
    _ = _mapping(canary)
    _ = _mutable_mapping(base)
    _ = _mutable_mapping(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mapping_view(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mapping_view(canary)  # pyright: ignore[reportArgumentType]


def _mapping(x: Mapping[Animal, Animal]) -> Mapping[Animal, Animal]:
    return x


def _mutable_mapping(
    x: MutableMapping[Animal, Animal],
) -> MutableMapping[Animal, Animal]:
    return x


def check_mapping_view_args(base: PyoMappingView, canary: MappingView) -> None:
    # pyrefly: ignore [bad-argument-type]
    _ = _iterable(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterable(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mapping_view(base)
    _ = _mapping_view(canary)


def check_items_view_args(
    base: PyoItemsView[Animal, Animal], canary: ItemsView[Animal, Animal]
) -> None:
    # pyrefly: ignore [bad-argument-type]
    _ = _iterable(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterable(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mapping_view(base)
    _ = _mapping_view(canary)
    _ = _items_view(base)
    _ = _items_view(canary)


def _items_view(x: ItemsView[Animal, Animal]) -> ItemsView[Animal, Animal]:
    return x


def check_values_view_args(
    base: PyoValuesView[Animal], canary: ValuesView[Animal]
) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)
    _ = _collection(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mapping_view(base)
    _ = _mapping_view(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _keys_view(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _keys_view(canary)  # pyright: ignore[reportArgumentType]
    _ = _values_view(base)
    _ = _values_view(canary)


def check_keys_view_args(base: PyoKeysView[Animal], canary: KeysView[Animal]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)
    _ = _collection(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mapping_view(base)
    _ = _mapping_view(canary)
    _ = _keys_view(base)
    _ = _keys_view(canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _values_view(base)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _ = _values_view(canary)  # pyright: ignore[reportArgumentType]


def _iterable(x: Iterable[Animal]) -> Iterable[Animal]:
    return x


def _iterator(x: Iterator[Animal]) -> Iterator[Animal]:
    return x


def _sized(x: Sized) -> Sized:
    return x


def _container(x: Container[Animal]) -> Container[Animal]:
    return x


def _reversible(x: Reversible[Animal]) -> Reversible[Animal]:
    return x


def _collection(x: Collection[Animal]) -> Collection[Animal]:
    return x


def _sequence(x: Sequence[Animal]) -> Sequence[Animal]:
    return x


def _mutable_sequence(x: MutableSequence[Dog]) -> MutableSequence[Dog]:
    return x


def _mapping_view(x: MappingView) -> MappingView:
    return x


def _keys_view(x: KeysView[Animal]) -> KeysView[Animal]:
    return x


def _values_view(x: ValuesView[Animal]) -> ValuesView[Animal]:
    return x


type EntryData = list[tuple[object, tuple[str, ...]]]


def covariance_pyomapping(data: EntryData) -> None:
    _: Dict[object, Sequence[object]] = assert_type(
        Dict(data), Dict[object, Sequence[object]]
    )


def check_chain_covariance[T, S](base: Iterable[T], *others: Iterable[S]) -> None:
    from pyochain import Iter

    _ = assert_type(Iter(base).chain(*others), PyoIterator[T | S])
    _ = assert_type(itertools.chain(base, *others), itertools.chain[T | S])


def check_dict() -> None:
    _ = assert_type(Dict({"a": 1, "b": 2}), Dict[str, int])
    # Avoid automatic literal inference
    data = [("a", 1), ("b", 2)]
    _ = assert_type(Dict(data), Dict[str, int])
    _ = assert_type(Dict(a=1, b=2), Dict[str, int])
    _ = assert_type(Dict({"a": 1}, b=2), Dict[str, int])


def _identity[T](x: T) -> T:
    return x
