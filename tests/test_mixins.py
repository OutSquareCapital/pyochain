import pytest

from pyochain import Dict, Range, Seq, Set, SetMut, Vec
from pyochain.abc import PyoSized
from pyochain.collections import Deque, SortedDict

DATA = Seq(1, 2, 3)
DATA_MAP = DATA.iter().map(str).enumerate().collect(Dict)
PARAMS_THEN_SOME: Vec[PyoSized] = Vec.wrap([
    DATA,
    Seq[int](()),
    DATA_MAP,
    Dict[str, int](()),
    Vec(DATA),
    Vec[int](()),
    Set(DATA),
    Set(()),
    SetMut(DATA),
    SetMut(()),
    DATA_MAP.keys(),
    DATA_MAP.values(),
    DATA_MAP.items(),
    Range(1, 4),
    Range(0, 0),
    SortedDict({"a": 1, "b": 2, "c": 3}),
    SortedDict[str, int](()),
    Deque(DATA),
    Deque[int](()),
])


@pytest.mark.parametrize(
    "data", PARAMS_THEN_SOME, ids=PARAMS_THEN_SOME.iter().map(repr).collect(tuple)
)
def test_then_some(data: PyoSized) -> None:
    assert data.then_some().is_some() if data.len() > 0 else data.then_some().is_none()


def test_then_some_peekable() -> None:
    assert DATA.iter().peekable().then_some().is_some()
