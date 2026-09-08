use std::ops::ControlFlow;

pub trait TryIterator: Iterator {
    /// Returns the first non-`None` value produced by `f`, or `None` if `f` returns `None` for every item.
    ///
    /// Iteration stops when `f` returns `Some` or an error.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `f`.
    fn try_find_map<B, E, F>(&mut self, mut f: F) -> Result<Option<B>, E>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Result<Option<B>, E>,
    {
        self.try_fold((), |(), item| match f(item) {
            Ok(Some(value)) => ControlFlow::Break(Ok(value)),
            Ok(None) => ControlFlow::Continue(()),
            Err(error) => ControlFlow::Break(Err(error)),
        })
        .break_value()
        .transpose()
    }
    /// Faillible version of `Iterator::all`.
    /// # Errors
    ///
    /// Returns the first error produced by the closure.
    fn try_all<E, F>(&mut self, f: F) -> Result<bool, E>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Result<bool, E>,
    {
        self.map(f)
            .find_map(|x| match x {
                Ok(true) => None,
                Ok(false) => Some(Ok(false)),
                Err(e) => Some(Err(e)),
            })
            .unwrap_or(Ok(true))
    }
    /// Faillible version of `Iterator::any`.
    /// # Errors
    ///
    /// Returns the first error produced by the closure.
    fn try_any<E, F>(&mut self, f: F) -> Result<bool, E>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Result<bool, E>,
    {
        self.map(f)
            .find_map(|x| match x {
                Ok(true) => Some(Ok(true)),
                Ok(false) => None,
                Err(e) => Some(Err(e)),
            })
            .unwrap_or(Ok(false))
    }
}

impl<I: Iterator> TryIterator for I {}
