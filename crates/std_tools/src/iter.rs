use std::ops::ControlFlow;

pub trait TryIterator: Iterator {
    /// Returns the first non-`None` value produced by `f`, or `None` if `f` returns `None` for every item.
    ///
    /// Iteration stops when `f` returns `Some` or an error.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by `f`.
    #[inline]
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
    #[inline]
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
    #[inline]
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
    /// Equivalent of `collect::<Result<Vec<T>, E>>()`, preallocated from the lower bound of `size_hint`.
    ///
    /// `std` collects a `Result` through an adapter whose lower bound is always 0, since it can stop at the first error.\
    /// The `Vec` then starts at a tiny capacity and grows by doubling, asking `size_hint` again on every growth.\
    /// This asks it once, which matters when it's costly to compute, e.g. `__length_hint__` on a Python iterator.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by the `Iterator`.
    #[inline]
    fn try_collect_vec_with_capacity<T, E>(mut self) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
    {
        let mut vec = Vec::with_capacity(self.size_hint().0);
        self.try_for_each(|item| item.map(|value| vec.push(value)))
            .map(|()| vec)
    }
}

impl<I: Iterator> TryIterator for I {}
