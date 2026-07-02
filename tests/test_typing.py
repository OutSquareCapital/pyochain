from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pyochain import (
    NONE,
    Err,
    Iter,
    Null,
    Ok,
    Option,
    Range,
    Result,
    Seq,
    Set,
    Some,
    option,
)

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Container,
        Iterable,
        Iterator,
        MutableSequence,
        Reversible,
        Sequence,
        Sized,
    )

    from pyochain import Peekable
    from pyochain.abc import (
        PyoCollection,
        PyoContainer,
        PyoIterable,
        PyoIterator,
        PyoMutableSequence,
        PyoReversible,
        PyoSequence,
        PyoSet,
        PyoSized,
    )


@dataclass
class Animal:
    pass


@dataclass
class Dog(Animal):
    pass


type AnimalLit = Literal["dog", "cat"]


def check_covariance() -> None:
    base: PyoIterable[Dog] = Iter(())
    opt: Option[Dog] = Some(Dog())
    res: Result[Dog, str] = Ok(Dog())
    _abc_iterable: PyoIterable[Animal] = base
    _abc_iterator: PyoIterator[Animal] = base
    _abc_collection: PyoCollection[Animal] = base.collect(Seq)
    _abc_sequence: PyoSequence[Animal] = base.collect(Seq)
    _concrete_iterator: Iter[Animal] = base
    _peekable_iterator: Peekable[Animal] = base.peekable()
    _abc_set_immutable: PyoSet[Animal] = base.collect(Set)
    _seq_immutable: Seq[Animal] = base.collect(Seq)
    _set_immutable: Set[Animal] = base.collect(Set)
    _as_opt: Option[Animal] = opt
    _as_res: Result[Animal, str] = res


def _get_cat() -> AnimalLit | None:
    return "cat"


def _value(x: Animal) -> Animal:
    return x


def _literal(x: AnimalLit) -> AnimalLit:
    return x


def _iterable(x: Iterable[Animal]) -> Iterable[Animal]:
    return x


def _iterator(x: Iterator[Animal]) -> Iterator[Animal]:
    return x


def _sized(x: Sized) -> Sized:
    return x


def _reversible(x: Reversible[Animal]) -> Reversible[Animal]:
    return x


def _container(x: Container[Animal]) -> Container[Animal]:
    return x


def _collection(x: Collection[Animal]) -> Collection[Animal]:
    return x


def _sequence(x: Sequence[Animal]) -> Sequence[Animal]:
    return x


def _mutable_sequence(x: MutableSequence[Dog]) -> MutableSequence[Dog]:
    return x


def check_option_basic() -> None:
    base = Some(Dog())
    canary: Dog | None = base.unwrap_or_none()
    _ = base.map(_value)
    if canary is not None:
        _ = _value(canary)


def check_option_transpose() -> None:
    _a: Result[Option[int], int] = Some(Ok(10)).transpose()
    _b: Result[Option[int], int] = Some(Err(10)).transpose()
    _c: Result[Option[int], int] = NONE.transpose()


def check_option_literal() -> None:  # noqa: C901
    lit = _get_cat()
    # Inferred as Option[str]
    opt_infered = option(lit)
    opt_casted: Option[AnimalLit] = option(lit)
    # Inferred as tuple[AnimalLit | None]
    canary_infered_tup = (lit,)
    # Inferred as list[str | None]
    canary_infered_list = [lit]
    canary_casted_list: list[AnimalLit | None] = [lit]
    # TODO: check if the literal inference for tuple but not for option nor list is a variance issue for option, or a special casing for tuple.
    _ = opt_infered.map(_literal)  # pyright: ignore[reportArgumentType]
    _ = opt_casted.map(_literal)
    _ = _literal(canary_infered_list[0]) if canary_infered_list[0] is not None else None  # pyright: ignore[reportArgumentType]
    _ = _literal(canary_casted_list[0]) if canary_casted_list[0] is not None else None
    _ = _literal(canary_infered_tup[0]) if canary_infered_tup[0] is not None else None

    match lit:
        case "dog":
            pass
        case "cat":
            pass
        case None:
            pass
    match canary_infered_tup:
        case ("dog",):
            pass
        case ("cat",):
            pass
        case (None,):
            pass
    # Here it doesn't work due to invariance of list (I think).
    match canary_casted_list:  # pyright: ignore[reportMatchNotExhaustive]
        case ["dog"]:
            pass
        case ["cat"]:
            pass
        case [None]:
            pass
    # But here it's an issue: Literals aren't handled for type unions, even if both members are covariant.
    match opt_casted:  # pyright: ignore[reportMatchNotExhaustive]
        case Some("dog"):
            pass
        case Some("cat"):
            pass
        case Null():
            pass


def check_result_basic() -> None:
    ok = Ok(Dog())
    err = Err(Dog())
    _a: Result[Animal, Animal] = ok.map(lambda x: x).map_err(_value)
    _b: Result[Animal, Animal] = ok.map_err(lambda x: x).map(_value)  # pyright: ignore[reportAny]
    _c: Result[Animal, Animal] = err.map(lambda x: x).map_err(_value)  # pyright: ignore[reportAny]
    # BUG: This should fail
    _d: Result[Animal, Animal] = err.map_err(lambda x: x).map(_value)


def check_result_transpose() -> None:
    """The error is expected.

    Rust equivalent (won't compile):
    ```rust
    fn check_result_transpose() -> () {
    let _a: Option<Result<u32, i32>> = Ok(Some(10)).transpose();
    let _b: Option<Result<i32, i32>> = Err(Some(10)).transpose();
    let _c: Option<Result<i32, i32>> = Ok(None).transpose();
    let _d: Option<Result<i32, i32>> = Err(None).transpose();
    }
    ```
    """
    _a: Option[Result[int, int]] = Ok(Some(10)).transpose()
    _b: Option[Result[int, int]] = Err(Some(10)).transpose()  # pyright: ignore[reportAssignmentType]
    _c: Option[Result[int, int]] = Ok(NONE).transpose()
    _d: Option[Result[int, int]] = Err(NONE).transpose()  # pyright: ignore[reportAssignmentType]


def check_option_flatten() -> None:
    _a: Option[int] = Some(Some(10)).flatten()
    _b: Option[int] = Some(NONE).flatten()
    _c: Option[int] = NONE.flatten()


def check_result_flatten() -> None:
    _a: Result[int, str] = Ok(Ok(10)).flatten()
    _b: Result[int, str] = Ok(Err("error")).flatten()
    _c: Result[int, str] = Err("error").flatten()
    # BUG: This should fail
    _d: Result[int, str] = Err(Err("error")).flatten()


def check_and_then() -> None:
    """The last case failing is expected.

    Rust equivalent (won't compile):
    ```rust
    fn test_flatten() {
    let _a: Result<i32, &str> = Ok(Ok(10)).and_then(|x| x);
    let _b: Result<i32, &str> = Ok(Err("error")).and_then(|x| x);
    let _c: Result<i32, &str> = Err("error").and_then(|x| x);
    let _d: Result<i32, &str> = Err(Err("error")).and_then(|x| x);
    }
    ```
    """
    _a: Result[int, str] = Ok(Ok(10)).and_then(lambda x: x)
    _b: Result[int, str] = Ok(Err("error")).and_then(lambda x: x)
    _c: Result[int, str] = Err("error").and_then(lambda x: x)  # pyright: ignore[reportAny]
    _d: Result[int, str] = Err(Err("error")).and_then(lambda x: x)  # pyright: ignore[reportAssignmentType, reportAny]


def test_iter_flatten() -> None:
    nested: PyoIterator[PyoIterator[PyoIterator[list[int]]]] = (
        Range(0, 3)
        .iter()
        .map(
            lambda x: (
                Range(0, x)
                .iter()
                .map(lambda y: Range(0, y).iter().map(lambda z: Range(0, z).pipe(list)))
            )
        )
    )
    one: PyoIterator[PyoIterator[list[int]]] = nested.flatten()
    two: PyoIterator[list[int]] = one.flatten()
    ok: PyoIterator[int] = two.flatten()
    # Expected to fail
    _fail = ok.flatten()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]


def check_iterable_args(base: PyoIterable[Dog], canary: Iterable[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_iterator_args(base: PyoIterator[Dog], canary: Iterator[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)
    _ = _iterator(canary)
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_sized_args(base: PyoSized, canary: Sized) -> None:
    _ = _iterable(base)  # pyright: ignore[reportArgumentType]
    _ = _iterable(canary)  # pyright: ignore[reportArgumentType]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_reversible_args(base: PyoReversible[Dog], canary: Reversible[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_container_args(base: PyoContainer[Animal], canary: Container[Animal]) -> None:
    _ = _iterable(base)  # pyright: ignore[reportArgumentType]
    _ = _iterable(canary)  # pyright: ignore[reportArgumentType]
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_collection_args(base: PyoCollection[Dog], canary: Collection[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_sequence_args(base: PyoSequence[Dog], canary: Sequence[Dog]) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
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
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_mutable_sequence_args(
    base: PyoMutableSequence[Dog], canary: MutableSequence[Dog]
) -> None:
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
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
