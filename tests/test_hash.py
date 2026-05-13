from dataclasses import dataclass, field

from pyochain import NONE, Err, Ok, Option, Result, Some


def test_option_variant_hash_matches_equality_contract() -> None:
    assert Some(42) != 42
    assert hash(Some(42)) == hash(Some(42))
    assert hash(NONE) == hash(None)


def test_result_variants_are_hashable() -> None:
    assert hash(Ok(42)) == hash(Ok(42))
    assert hash(Err("boom")) == hash(Err("boom"))
    assert hash(Ok(42)) != hash(Err(42))


def test_frozen_dataclass_accepts_variant_fields() -> None:
    @dataclass(frozen=True)
    class Foo:
        maybe: Option[int] = NONE
        outcome: Result[int, str] = field(default=Ok(42))

    assert isinstance(hash(Foo(Some(42), Ok(42))), int)
    assert isinstance(hash(Foo(NONE, Err("boom"))), int)
