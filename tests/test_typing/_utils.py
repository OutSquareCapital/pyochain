from __future__ import annotations

from typing import Literal


class Animal:
    def as_parent(self) -> Animal:
        return self


class Dog(Animal): ...


type LitDog = Literal["dog"]
type LitCat = Literal["cat"]
type AnimalLit = Literal[LitDog, LitCat]


def identity[T](x: T) -> T:
    return x
