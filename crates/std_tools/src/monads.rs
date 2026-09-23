use either::Either;

#[allow(unused)]
pub trait OptionExt<T, R, E> {
    fn map_transpose(self, func: impl FnOnce(T) -> R) -> Result<Option<R>, E>;
    fn and_then_transpose(self, func: impl FnOnce(T) -> Result<R, E>) -> Result<Option<R>, E>;
}
impl<T, R, E> OptionExt<T, R, E> for Option<Result<T, E>> {
    /// Transforms an `Option<Result<T, E>>` into a `Result<Option<R>, E>` by applying a function to the `Ok` value if it exists.\
    /// If the `Option` is `None`, it returns `Ok(None)`.\
    /// If the `Option` is `Some(Err(e))`, it returns `Err(e)`.\
    /// If the `Option` is `Some(Ok(item))`, it applies the function to `item` and wraps the result in `Some`.\
    /// This can be useful to replace nested `transpose()` and `map()` calls with a single method call, improving readability.
    #[inline]
    fn map_transpose(self, func: impl FnOnce(T) -> R) -> Result<Option<R>, E> {
        match self {
            Some(Ok(item)) => Ok(Some(func(item))),
            None => Ok(None),
            Some(Err(e)) => Err(e),
        }
    }
    /// Similar to `map_transpose`, but the function returns a `Result<R, E>`.\
    /// Allows for chaining operations that may fail, while still handling the `Option` case.
    #[inline]
    fn and_then_transpose(self, func: impl FnOnce(T) -> Result<R, E>) -> Result<Option<R>, E> {
        match self {
            Some(Ok(item)) => func(item).map(Some),
            None => Ok(None),
            Some(Err(e)) => Err(e),
        }
    }
}
pub trait ResultExt<L, R, E> {
    fn map_left<T>(self, func: impl FnOnce(L) -> T) -> Result<Either<T, R>, E>;
    fn map_right<T>(self, func: impl FnOnce(R) -> T) -> Result<Either<L, T>, E>;
    fn and_then_left<T>(self, func: impl FnOnce(L) -> Result<T, E>) -> Result<Either<T, R>, E>;
    fn and_then_right<T>(self, func: impl FnOnce(R) -> Result<T, E>) -> Result<Either<L, T>, E>;
}
impl<L, R, E> ResultExt<L, R, E> for Result<Either<L, R>, E> {
    #[inline]
    fn map_left<T>(self, func: impl FnOnce(L) -> T) -> Result<Either<T, R>, E> {
        match self {
            Ok(Either::Left(l)) => Ok(Either::Left(func(l))),
            Ok(Either::Right(r)) => Ok(Either::Right(r)),
            Err(e) => Err(e),
        }
    }
    #[inline]
    fn map_right<T>(self, func: impl FnOnce(R) -> T) -> Result<Either<L, T>, E> {
        match self {
            Ok(Either::Left(l)) => Ok(Either::Left(l)),
            Ok(Either::Right(r)) => Ok(Either::Right(func(r))),
            Err(e) => Err(e),
        }
    }
    #[inline]
    fn and_then_left<T>(self, func: impl FnOnce(L) -> Result<T, E>) -> Result<Either<T, R>, E> {
        match self {
            Ok(Either::Left(l)) => func(l).map(Either::Left),
            Ok(Either::Right(r)) => Ok(Either::Right(r)),
            Err(e) => Err(e),
        }
    }
    #[inline]
    fn and_then_right<T>(self, func: impl FnOnce(R) -> Result<T, E>) -> Result<Either<L, T>, E> {
        match self {
            Ok(Either::Left(l)) => Ok(Either::Left(l)),
            Ok(Either::Right(r)) => func(r).map(Either::Right),
            Err(e) => Err(e),
        }
    }
}
