import pytest

from pyochain import Dict

DATA = [("a", 1), ("b", 2)]


@pytest.mark.parametrize(
    "data",
    (Dict(dict(DATA)), Dict(DATA), Dict(*DATA), Dict(a=1, b=2)),
)
def test_constructor(data: Dict[str, int]) -> None:
    assert data == {"a": 1, "b": 2}
