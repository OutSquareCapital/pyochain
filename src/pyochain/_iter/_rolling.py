from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import rolling

from .._core import IterWrapper

if TYPE_CHECKING:
    from ._main import Iter


class BaseRolling[T](IterWrapper[T]):
    def rolling_mean(self, window_size: int) -> Iter[float]:
        """Compute the rolling mean.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[float]: An iterable of rolling mean values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3, 4, 5]).iter().rolling_mean(3).into(list)
        [2.0, 3.0, 4.0]

        ```

        """
        return self._lazy(rolling.Mean, window_size)

    def rolling_median(self, window_size: int) -> Iter[T]:
        """Compute the rolling median.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[T]: An iterable of rolling median values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 3, 2, 5, 4]).iter().rolling_median(3).into(list)
        [2, 3, 4]

        ```

        """
        return self._lazy(rolling.Median, window_size)

    def rolling_sum(self, window_size: int) -> Iter[T]:
        """Compute the rolling sum.

        Will return integers if the input is integers, floats if the input is floats or mixed.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[T]: An iterable of rolling sum values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1.0, 2, 3, 4, 5]).iter().rolling_sum(3).into(list)
        [6.0, 9.0, 12.0]

        ```

        """
        return self._lazy(rolling.Sum, window_size)

    def rolling_min(self, window_size: int) -> Iter[T]:
        """Compute the rolling minimum.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[T]: An iterable of rolling minimum values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 4, 1, 5, 9, 2]).iter().rolling_min(3).into(list)
        [1, 1, 1, 1, 2]

        ```

        """
        return self._lazy(rolling.Min, window_size)

    def rolling_max(self, window_size: int) -> Iter[T]:
        """Compute the rolling maximum.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[T]: An iterable of rolling maximum values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 4, 1, 5, 9, 2]).iter().rolling_max(3).into(list)
        [4, 4, 5, 9, 9]

        ```

        """
        return self._lazy(rolling.Max, window_size)

    def rolling_var(self, window_size: int) -> Iter[float]:
        """Compute the rolling variance.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[float]: An iterable of rolling variance values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 4, 1, 4]).iter().rolling_var(3).map(lambda x: round(x, 2)).into(list)
        [2.33, 2.33, 3.0]

        ```

        """
        return self._lazy(rolling.Var, window_size)

    def rolling_std(self, window_size: int) -> Iter[float]:
        """Compute the rolling standard deviation.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[float]: An iterable of rolling standard deviation values.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 4, 1, 4]).iter().rolling_std(3).map(lambda x: round(x, 2)).collect()
        Seq((1.53, 1.53, 1.73))

        ```

        """
        return self._lazy(rolling.Std, window_size)

    def rolling_kurtosis(self, window_size: int) -> Iter[float]:
        """Compute the rolling kurtosis.

        Args:
            window_size (int): Size of the rolling window. Must be at least 4.

        Returns:
            Iter[float]: An iterable of rolling kurtosis values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 4, 1, 4]).iter().rolling_kurtosis(4).into(list)
        [1.5, -3.901234567901234]

        ```

        """
        return self._lazy(rolling.Kurtosis, window_size)

    def rolling_skew(self, window_size: int) -> Iter[float]:
        """Compute the rolling skewness.

        Args:
            window_size (int): Size of the rolling window. Must be at least 3.

        Returns:
            Iter[float]: An iterable of rolling skewness values.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 4, 1, 4]).iter().rolling_skew(3).map(lambda x: round(x, 2)).collect()
        Seq((0.94, 0.94, -1.73))

        ```

        """
        return self._lazy(rolling.Skew, window_size)

    def rolling_all(self, window_size: int) -> Iter[bool]:
        """Compute whether all values in the window evaluate to True.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[bool]: An iterable of rolling all boolean values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([True, True, False, True, True]).iter().rolling_all(2).into(list)
        [True, False, False, True]

        ```

        """
        return self._lazy(rolling.All, window_size)

    def rolling_any(self, window_size: int) -> Iter[bool]:
        """Compute whether any value in the window evaluates to True.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[bool]: An iterable of rolling any values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([True, True, False, True, True]).iter().rolling_any(2).into(list)
        [True, True, True, True]

        ```

        """
        return self._lazy(rolling.Any, window_size)

    def rolling_product(self, window_size: int) -> Iter[float]:
        """Compute the rolling product.

        Args:
            window_size (int): Size of the rolling window.

        Returns:
            Iter[float]: An iterable of rolling product values.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3, 4, 5]).iter().rolling_product(3).into(list)
        [6.0, 24.0, 60.0]

        ```

        """
        return self._lazy(rolling.Product, window_size)

    def rolling_apply[R](
        self,
        func: Callable[[Iterable[T]], R],
        window_size: int,
    ) -> Iter[R]:
        """Apply a custom function to each rolling window.

        The function should accept an iterable and return a single value.

        Args:
            func (Callable[[Iterable[T]], R]): Function to apply to each rolling window.
            window_size (int): Size of the rolling window.

        Returns:
            Iter[R]: An iterable of results from applying the function.
        ```python
        >>> import pyochain as pc
        >>> def range_func(window):
        ...     return max(window) - min(window)
        >>> pc.Seq([1, 3, 2, 5, 4]).iter().rolling_apply(range_func, 3).into(list)
        [2, 3, 3]

        ```

        """
        return self._lazy(rolling.Apply, window_size, "fixed", func)

    def rolling_apply_pairwise[R](
        self,
        other: Iterable[T],
        func: Callable[[T, T], R],
        window_size: int,
    ) -> Iter[R]:
        """Apply a custom pairwise function to each rolling window of size 2.

        The function should accept two arguments and return a single value.

        Args:
            other (Iterable[T]): Second iterable to apply the pairwise function.
            func (Callable[[T, T], R]): Function to apply to each pair of elements.
            window_size (int): Size of the rolling window.

        Returns:
            Iter[R]: An iterable of results from applying the pairwise function.
        ```python
        >>> import pyochain as pc
        >>> from statistics import correlation as corr
        >>> seq_1 = [1, 2, 3, 4, 5]
        >>> seq_2 = [1, 2, 3, 2, 1]
        >>> pc.Seq(seq_1).iter().rolling_apply_pairwise(seq_2, corr, 3).into(list)
        [1.0, 0.0, -1.0]

        ```

        """
        return self._lazy(rolling.ApplyPairwise, other, window_size, func)
