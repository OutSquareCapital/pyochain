from collections.abc import Iterable, Iterator


class _BaseImpl:
    def __init__(self) -> None:
        self._data: list[int] = [1, 2, 3]


class ImplIter(_BaseImpl):
    def __iter__(self) -> Iterator[int]:
        return iter(self._data)


class ImplSized(_BaseImpl):
    def __len__(self) -> int:
        return len(self._data)


class ImplRev(ImplIter):
    def __reversed__(self) -> Iterator[int]:
        return reversed(self._data)


class ImplContainer(_BaseImpl):
    def __contains__(self, item: int) -> bool:
        return item in self._data


class ImplCollection(ImplSized, ImplContainer, ImplIter):
    """Works to implement both `Collection` and `Set` ABCs as a standlone."""

    def __init__(self, it: Iterable[int] | None = None) -> None:
        self._data: list[int] = [1, 2, 3] if it is None else list(dict.fromkeys(it))


class ImplSequence(ImplSized):
    def __getitem__(self, index: int) -> int:
        return self._data[index]


class ImplMapping(ImplSequence, ImplIter):
    def __init__(self) -> None:
        self._data: dict[int, int] = {i: i * 10 for i in [0, 1, 2]}  # pyright: ignore[reportIncompatibleVariableOverride]


class ImplMutableMapping(ImplMapping):
    def __setitem__(self, key: int, value: int) -> None:
        self._data[key] = value

    def __delitem__(self, key: int) -> None:
        del self._data[key]


class ImplMutableSequence(ImplSequence):
    def __setitem__(self, index: int, value: int) -> None:
        self._data[index] = value

    def __delitem__(self, index: int) -> None:
        del self._data[index]

    def insert(self, index: int, value: int) -> None:
        self._data.insert(index, value)


def assert_iter_eq(a: Iterable[object], b: Iterable[object]) -> None:
    assert tuple(a) == tuple(b)
