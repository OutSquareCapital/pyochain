from pyochain import Seq


def test_last() -> None:
    data = Seq((1, 2, 3, 4))
    assert data.iter().last() == 4


def test_last_unique() -> None:
    data = Seq((1, 2, 1, 3))
    assert data.iter().unique().last() == 3


def test_last_unique_by() -> None:
    data = Seq(("cat", "mouse", "dog", "hen"))
    assert data.iter().unique_by(len).last() == "mouse"


def test_length_unique() -> None:
    data = Seq((1, 2, 1, 3))
    assert data.iter().unique().length() == 3


def test_length_unique_by() -> None:
    data = Seq(("cat", "mouse", "dog", "hen"))
    assert data.iter().unique_by(len).length() == 2
