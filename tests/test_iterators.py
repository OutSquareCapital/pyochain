import pytest

from pyochain import Iter


@pytest.mark.parametrize("length", (-1, 0, 1))
def test_map_windows_fail(length: int) -> None:
    match length:
        case -1:
            _with_err(length, OverflowError)
        case 0:
            _with_err(length, ValueError)
        case _:
            vals = Iter.repeat(1).take(5).map_windows(length, sum).sum()
            assert vals == 5


def _with_err(length: int, err: type[Exception]) -> None:
    it = Iter.once(1)
    with pytest.raises(err):
        _ = it.map_windows(length, len)
    with pytest.raises(err):
        _ = it.map_windows_star(length, Iter.of)
