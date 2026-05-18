from pyochain import NONE, Null


def test_none_identity() -> None:
    assert Null() is NONE
    assert Null() is Null()
