"""Benchmarks for pyochain package - benchs.py."""

from dataclasses import dataclass

import pyochain as pc

from ._registery import bench

# Helper functions
# ------------------------------------------------------------


def _args_fn(x: int, y: int) -> int:
    return x + y


def _kwargs_fn(x: int, y: int, *, z: int) -> int:
    return x + y + z


def _kwargs_no_args_fn(x: int, *, z: int) -> int:
    return x + z


def _args_fn_star(a: int, b: int, c: int) -> int:
    return a + b + c


def _kwargs_fn_star(a: int, b: int, c: int, *, d: int) -> int:
    return a + b + c + d


def _kwargs_no_args_fn_star(a: int, b: int, *, c: int) -> int:
    return a + b + c


@dataclass(slots=True)
class Point:  # noqa: D101, PLW1641
    x: int
    y: int

    def __eq__(self, other: object) -> bool:
        """Equality comparison for Point dataclass."""
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y


# Benchmark classes
# ------------------------------------------------------------


class AllEqual:
    """Benchmark all_equal with different data types."""

    @bench(gen=lambda size: size.map(lambda _: 42).collect())
    @staticmethod
    def with_int(data: pc.Seq[int]) -> object:
        """Benchmark all_equal with int data."""
        return data.iter().all_equal()

    @bench(gen=lambda size: size.map(lambda _: 3.14159).collect())
    @staticmethod
    def with_float(data: pc.Seq[float]) -> object:
        """Benchmark all_equal with float data."""
        return data.iter().all_equal()

    @bench(gen=lambda size: size.map(lambda _: True).collect())
    @staticmethod
    def with_bool(data: pc.Seq[bool]) -> object:
        """Benchmark all_equal with bool data."""
        return data.iter().all_equal()

    @bench(gen=lambda size: size.map(lambda _: "hello world").collect())
    @staticmethod
    def with_str(data: pc.Seq[str]) -> object:
        """Benchmark all_equal with str data."""
        return data.iter().all_equal()

    @bench(gen=lambda size: size.map(lambda _: Point(1, 2)).collect())
    @staticmethod
    def with_dataclass(data: pc.Seq[Point]) -> object:
        """Benchmark all_equal with custom dataclass (uses __eq__)."""
        return data.iter().all_equal()


class ForEach:
    """Benchmark `for_each` with different argument types."""

    @bench()
    @staticmethod
    def for_each(data: pc.Seq[int]) -> object:
        """Benchmark for_each implementation."""
        return data.iter().for_each(lambda x: x * 2)

    @bench()
    @staticmethod
    def for_each_args(data: pc.Seq[int]) -> object:
        """Benchmark for_each implementation."""
        return data.iter().for_each(_args_fn, 20)

    @bench()
    @staticmethod
    def for_each_kwargs(data: pc.Seq[int]) -> object:
        """Benchmark for_each implementation."""
        return data.iter().for_each(_kwargs_fn, 20, z=30)

    @bench()
    @staticmethod
    def for_each_kwargs_no_args(data: pc.Seq[int]) -> object:
        """Benchmark for_each implementation."""
        return data.iter().for_each(_kwargs_no_args_fn, z=50)


class ForEachStar:
    """Benchmark `for_each_star` with different argument types."""

    @bench(gen=lambda size: size.map(lambda x: (x, x)).collect())
    @staticmethod
    def for_each_star(data: pc.Seq[tuple[int, int]]) -> object:
        """Benchmark for_each_star implementation."""
        return data.iter().for_each_star(lambda x, y: x + y)

    @bench(gen=lambda size: size.map(lambda x: (x, x)).collect())
    @staticmethod
    def for_each_star_args(data: pc.Seq[tuple[int, int]]) -> object:
        """Benchmark for_each implementation."""
        return data.iter().for_each_star(_args_fn_star, 20)

    @bench(gen=lambda size: size.map(lambda x: (x, x)).collect())
    @staticmethod
    def for_each_star_kwargs(data: pc.Seq[tuple[int, int]]) -> object:
        """Benchmark for_each implementation."""
        return data.iter().for_each_star(_kwargs_fn_star, 20, d=30)

    @bench(gen=lambda size: size.map(lambda x: (x, x)).collect())
    @staticmethod
    def for_each_star_kwargs_no_args(data: pc.Seq[tuple[int, int]]) -> object:
        """Benchmark for_each implementation."""
        return data.iter().for_each_star(_kwargs_no_args_fn_star, c=50)
