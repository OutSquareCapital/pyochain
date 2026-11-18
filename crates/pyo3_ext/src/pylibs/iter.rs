//! `Iterator` functions from builtins and itertools, wrapped in a newtype to align with an `Iterator`-like API.
use std::borrow::Borrow;

use crate::{
    args::{CallConcat, IntoArgs},
    pylibs::registery::{REDUCE, builtins, itertools},
    tuple,
};
use derive_more::{Constructor, IntoIterator};
use pyo3::{
    intern,
    prelude::*,
    types::{PyBool, PyDict, PyInt, PyIterator, PyList, PyNone, PyTuple},
};
use tap::prelude::*;
type ChainedIter<'py> = PyResult<NativeIter<Bound<'py, PyIterator>>>;

#[derive(Constructor, IntoIterator)]
pub struct NativeIter<T>(T);
impl<T> NativeIter<T> {
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> T {
        self.0
    }
}
impl<'py, T> AsRef<Bound<'py, PyIterator>> for NativeIter<T>
where
    T: Borrow<Bound<'py, PyIterator>>,
{
    fn as_ref(&self) -> &Bound<'py, PyIterator> {
        self.0.borrow()
    }
}
impl<'py, T> NativeIter<T>
where
    T: Borrow<Bound<'py, PyIterator>>,
{
    #[must_use]
    pub fn py(&self) -> Python<'py> {
        self.as_ref().py()
    }
    #[inline(always)]
    pub fn all(&self) -> PyResult<Bound<'py, PyBool>> {
        builtins::ALL
            .import(self.py())
            .call1((self.as_ref(),))
            .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
    }
    #[inline(always)]
    pub fn any(&self) -> PyResult<Bound<'py, PyBool>> {
        builtins::ANY
            .import(self.py())
            .call1((self.as_ref(),))
            .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
    }
    #[inline(always)]
    pub fn enumerate(self, start: usize) -> ChainedIter<'py> {
        builtins::ENUMERATE
            .import(self.py())
            .call1((self.as_ref(), start))
            .map(rewrap)
    }

    #[inline(always)]
    pub fn filter(self, func: Option<&Bound<'py, PyAny>>) -> ChainedIter<'py> {
        builtins::FILTER
            .import(self.py())
            .call1((func, self.as_ref()))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn max(&self) -> PyResult<Bound<'py, PyAny>> {
        builtins::MAX.import(self.py()).call1((self.as_ref(),))
    }

    #[inline(always)]
    pub fn max_by(self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(self.py());
        kwargs.set_item(intern!(self.py(), "key"), key)?;
        builtins::MAX
            .import(self.py())
            .call((self.as_ref(),), Some(&kwargs))
    }
    #[inline(always)]
    pub fn min(&self) -> PyResult<Bound<'py, PyAny>> {
        builtins::MIN.import(self.py()).call1((self.as_ref(),))
    }

    #[inline(always)]
    pub fn min_by(self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(self.py());
        kwargs.set_item(intern!(self.py(), "key"), key)?;
        builtins::MIN
            .import(self.py())
            .call((self.as_ref(),), Some(&kwargs))
    }
    #[inline(always)]
    pub fn sum(&self, start: &i32) -> PyResult<Bound<'py, PyAny>> {
        builtins::SUM
            .import(self.py())
            .call1((self.as_ref(), start))
    }
    #[inline(always)]
    pub fn map(self, func: &Bound<'py, PyAny>) -> ChainedIter<'py> {
        builtins::MAP
            .import(self.py())
            .call1((func, self.as_ref()))
            .map(rewrap)
    }

    #[inline(always)]
    pub fn sorted(
        self,
        reverse: bool,
        key: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let py = self.py();
        let func = builtins::SORTED.import(py);
        let it = (self.as_ref(),);
        // Micro-optimization to minimize PyDict::set_item calls, and use vector call if possible.
        match (reverse, key) {
            (false, None) => func.call1(it),
            (true, None) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "reverse"), true)?;
                func.call(it, Some(&kwargs))
            }
            (false, Some(key)) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "key"), key)?;
                func.call(it, Some(&kwargs))
            }
            (true, Some(key)) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "reverse"), true)?;
                kwargs.set_item(intern!(py, "key"), key)?;
                func.call(it, Some(&kwargs))
            }
        }
        .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
    }
    #[inline(always)]
    pub fn zip(self, others: &Bound<'py, PyTuple>, strict: bool) -> ChainedIter<'py> {
        let py = self.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "strict"), strict)?;
        builtins::ZIP
            .import(py)
            .call_concat((self.as_ref().as_any(), others), Some(&kwargs))
            .map(rewrap)
    }

    #[inline(always)]
    pub fn reduce(
        self,
        function: &Bound<'py, PyAny>,
        initial: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args = match initial {
            Some(initial) => tuple!(function, self.as_ref(), initial)?,
            None => tuple!(function, self.as_ref())?,
        };
        REDUCE.import(function.py()).call1(args)
    }
    // itertools

    #[inline(always)]
    pub fn tee(&self, n: usize) -> PyResult<Bound<'py, PyTuple>> {
        itertools::TEE
            .import(self.py())
            .call1((self.as_ref(), n))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyTuple>() })
    }
    #[inline(always)]
    pub fn take_while(&self, predicate: &Bound<'py, PyAny>) -> ChainedIter<'py> {
        itertools::TAKE_WHILE
            .import(self.py())
            .call1((predicate, self.as_ref()))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn drop_while(&self, predicate: &Bound<'py, PyAny>) -> ChainedIter<'py> {
        itertools::DROP_WHILE
            .import(self.py())
            .call1((predicate, self.as_ref()))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn accumulate(
        &self,
        func: Option<Bound<'py, PyAny>>,
        initial: Option<Bound<'py, PyAny>>,
    ) -> ChainedIter<'py> {
        let py = self.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "initial"), initial)?;
        itertools::ACCUMULATE
            .import(py)
            .call((self.as_ref(), func), Some(&kwargs))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn batched(&self, n: &Bound<'py, PyInt>, strict: &bool) -> ChainedIter<'py> {
        let py = self.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "strict"), strict)?;
        itertools::BATCHED
            .import(py)
            .call((self.as_ref(), n), Some(&kwargs))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn compress(&self, selectors: &Bound<'py, PyTuple>) -> ChainedIter<'py> {
        itertools::COMPRESS
            .import(self.py())
            .call1((self.as_ref(), selectors))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn cycle(&self) -> ChainedIter<'py> {
        itertools::CYCLE
            .import(self.py())
            .call1((self.as_ref(),))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn permutations(&self, r: Option<usize>) -> ChainedIter<'py> {
        itertools::PERMUTATIONS
            .import(self.py())
            .call1((self.as_ref(), r))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn combinations(&self, r: &Bound<'py, PyInt>) -> ChainedIter<'py> {
        itertools::COMBINATIONS
            .import(self.py())
            .call1((self.as_ref(), r))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn combinations_with_replacement(&self, r: &Bound<'py, PyInt>) -> ChainedIter<'py> {
        itertools::COMBINATIONS_WITH_REPLACEMENT
            .import(self.py())
            .call1((self.as_ref(), r))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn filter_false(&self, func: Option<Bound<'py, PyAny>>) -> ChainedIter<'py> {
        itertools::FILTER_FALSE
            .import(self.py())
            .call1((func, self.as_ref()))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn group_by(&self, key: Option<Bound<'py, PyAny>>) -> ChainedIter<'py> {
        itertools::GROUP_BY
            .import(self.py())
            .call1((self.as_ref(), key))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn pairwise(&self) -> ChainedIter<'py> {
        itertools::PAIRWISE
            .import(self.py())
            .call1((self.as_ref(),))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn map_star(&self, func: &Bound<'py, PyAny>) -> ChainedIter<'py> {
        itertools::MAP_STAR
            .import(self.py())
            .call1((func, self.as_ref()))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn nth(&self, n: usize) -> PyResult<Option<Bound<'py, PyAny>>> {
        itertools::ISLICE
            .import(self.py())
            .call1((self.as_ref(), n, n + 1))?
            .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
            .next()
            .transpose()
    }
    #[inline(always)]
    pub fn slice(
        &self,
        start: Option<&Bound<'py, PyInt>>,
        stop: Option<&Bound<'py, PyInt>>,
        step: Option<&Bound<'py, PyInt>>,
    ) -> ChainedIter<'py> {
        itertools::ISLICE
            .import(self.py())
            .call1((self.as_ref(), start, stop, step))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn skip(self, n: &Bound<'py, PyInt>) -> ChainedIter<'py> {
        let py = self.py();
        itertools::ISLICE
            .import(py)
            .call1((self.as_ref(), n, PyNone::get(py)))
            .map(rewrap)
    }

    #[inline(always)]
    pub fn zip_longest(&self, others: &Bound<'py, PyTuple>) -> ChainedIter<'py> {
        itertools::ZIP_LONGEST
            .import(self.py())
            .call_concat1((self.as_ref().as_any(), others))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn step_by(&self, step: &Bound<'py, PyInt>) -> ChainedIter<'py> {
        let py = self.py();
        itertools::ISLICE
            .import(py)
            .call1((self.as_ref(), 0, PyNone::get(py), step))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn take(&self, stop: &Bound<'py, PyInt>) -> ChainedIter<'py> {
        itertools::ISLICE
            .import(self.py())
            .call1((self.as_ref(), stop))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn chain<A: IntoArgs<'py>>(&self, others: A) -> ChainedIter<'py> {
        itertools::CHAIN
            .import(self.py())
            .call_concat1((self.as_ref().as_any(), others))
            .map(rewrap)
    }
    #[inline(always)]
    pub fn flatten(&self) -> ChainedIter<'py> {
        let py = self.py();
        itertools::CHAIN
            .import(py)
            .getattr(intern!(py, "from_iterable"))?
            .call1((self.as_ref(),))
            .map(rewrap)
    }

    #[inline(always)]
    pub fn product<A: IntoArgs<'py>>(&self, iterables: A, repeat: usize) -> ChainedIter<'py> {
        let py = self.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "repeat"), repeat)?;
        itertools::PRODUCT
            .import(py)
            .call_concat((self.as_ref().as_any(), iterables), Some(&kwargs))
            .map(rewrap)
    }

    /// first arg is a function, the rest is a variable number of iterables.
    #[inline(always)]
    pub fn map_with(
        &self,
        func: Bound<'py, PyAny>,
        iterables: &Bound<'py, PyTuple>,
    ) -> ChainedIter<'py> {
        builtins::MAP
            .import(self.py())
            .call_concat1((func, (self.as_ref().as_any(), iterables)))
            .map(rewrap)
    }
}

#[inline(always)]
pub fn count<'py>(py: Python<'py>, start: &i32, step: &i32) -> ChainedIter<'py> {
    itertools::COUNT.import(py).call1((start, step)).map(rewrap)
}

fn rewrap(it: Bound<'_, PyAny>) -> NativeIter<Bound<'_, PyIterator>> {
    unsafe { it.cast_into_unchecked::<PyIterator>() }.pipe(NativeIter::new)
}
