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
