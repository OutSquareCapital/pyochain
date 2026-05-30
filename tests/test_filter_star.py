from pyochain import Iter, Seq


def test_filter_star_preserves_non_tuple_iterables() -> None:
    data = ([0, "a"], [1, "b"], [2, "c"])

    def is_even_index(index: int, _item: str) -> bool:
        return index % 2 == 0

    assert Iter(data).filter_star(is_even_index).collect() == Seq(([0, "a"], [2, "c"]))
