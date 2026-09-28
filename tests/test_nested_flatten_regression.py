from pyochain import Iter


def test_flatten_nested_iterators_regression() -> None:
    nested = Iter([
        Iter([1, 2]),
        Iter([Iter([3, 4]), Iter([5])]),
        [6, 7],
    ])

    assert list(nested.flatten()) == [1, 2, 3, 4, 5, 6, 7]


def test_flat_map_works_with_nested_iterable_results() -> None:
    out = Iter([1, 2, 3]).flat_map(lambda x: Iter([x, x * 10]))

    assert list(out) == [1, 10, 2, 20, 3, 30]


def test_repeated_flatten_removes_all_levels() -> None:
    nested = Iter([
        Iter([Iter([1]), Iter([2])]),
        Iter([Iter([3, 4])]),
    ])

    assert list(nested.flatten().flatten()) == [1, 2, 3, 4]


def test_recursive_flatten_handles_three_levels() -> None:
    nested = Iter([
        Iter([1]),
        Iter([Iter([2]), Iter([Iter([3, 4]), Iter([5])])]),
        [[6], [7, 8]],
    ])

    assert list(nested.flatten()) == [1, 2, 3, 4, 5, 6, 7, 8]
