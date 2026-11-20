import pytest

from pyochain.collections import SortedList


def test_init_err() -> None:
    with pytest.raises(TypeError):
        _ = SortedList((object(), object()))  # pyright: ignore[reportArgumentType, reportUnknownVariableType]
