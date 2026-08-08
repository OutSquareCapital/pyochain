use std::sync::Mutex;

use super::traits::InnerSorted;
use crate::{abc, collections::InnerKeyLists, iterators, traits::PyoABC};
use pyo3::{PyClass, prelude::*};
use tap::Pipe;

pub trait SortedIterator<T: InnerSorted>: Sized {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>>;
    fn into_pyoiterator<'py, O: PyClass<BaseType = abc::PyoIterator>>(
        self,
        py: Python<'py>,
        base: O,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        abc::PyoIterator::build_init()
            .add_subclass(base)
            .pipe(|x| Bound::new(py, x))?
            .into_super()
            .pipe(Ok)
    }
}

struct IterIdxs {
    outer: usize,
    inner: usize,
}

pub struct IsliceBounds {
    pub min_pos: usize,
    pub max_pos: usize,
    pub min_idx: usize,
    pub max_idx: usize,
}
impl IsliceBounds {
    pub fn new(min_pos: usize, min_idx: usize, max_pos: usize, max_idx: usize) -> Self {
        Self {
            min_pos,
            min_idx,
            max_pos,
            max_idx,
        }
    }
}
pub enum SliceKind {
    Empty,
    MinEqMax,
    MinEqMaxRev,
    NextEqMax,
    NextEqMaxRev,
    MinLtMax,
    MinLtMaxRev,
}
impl SliceKind {
    #[inline]
    pub fn new(bounds: &IsliceBounds, reverse: bool) -> Self {
        let next_pos = bounds.min_pos + 1;
        if bounds.min_pos > bounds.max_pos {
            return Self::Empty;
        }

        if bounds.min_pos == bounds.max_pos {
            if reverse {
                return Self::MinEqMaxRev;
            }
            return Self::MinEqMax;
        }

        if next_pos == bounds.max_pos {
            if reverse {
                return Self::NextEqMaxRev;
            }
            return Self::NextEqMax;
        }

        if reverse {
            return Self::MinLtMaxRev;
        }

        Self::MinLtMax
    }
    pub fn into_iterator<'py, T: InnerSorted>(
        self,
        py: Python<'py>,
        slf: Py<T>,
        bounds: IsliceBounds,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        match self {
            Self::Empty => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Self::MinEqMax => MinEqMaxIter::new(slf, bounds).into_pyoiterator(py),
            Self::MinEqMaxRev => MinEqMaxIterRev::new(slf, bounds).into_pyoiterator(py),
            Self::NextEqMax => NextEqMaxIter::new(slf, bounds).into_pyoiterator(py),
            Self::NextEqMaxRev => NextEqMaxIterRev::new(slf, bounds).into_pyoiterator(py),
            Self::MinLtMax => MinLtMaxIter::new(slf, bounds).into_pyoiterator(py),
            Self::MinLtMaxRev => MinLtMaxIterRev::new(slf, bounds).into_pyoiterator(py),
        }
    }
}
#[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
pub struct SortedIter {
    inner: Mutex<SortedIterInner<InnerKeyLists>>,
}
#[pymethods]
impl SortedIter {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner
            .lock()
            .expect("Failed to lock SortedIterInner")
            .next(py)
    }
}
#[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
pub struct SortedIterKey {
    inner: Mutex<SortedIterInner<InnerKeyLists>>,
}
#[pymethods]
impl SortedIterKey {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner
            .lock()
            .expect("Failed to lock SortedIterInner")
            .next(py)
    }
}
/// ref: `self.lists.iter().flatten()` for `__iter__`
struct SortedIterInner<T: InnerSorted> {
    owner: Py<T>,
    cursor: IterIdxs,
    done: bool,
}

impl<T: InnerSorted> SortedIterInner<T> {
    pub fn new(owner: Py<T>) -> Self {
        Self {
            owner,
            cursor: IterIdxs { outer: 0, inner: 0 },
            done: false,
        }
    }
}
impl<T: InnerSorted> SortedIterator<T> for SortedIterInner<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.done {
            None
        } else {
            let lists = self.owner.get().get_lists();
            loop {
                match lists.get(self.cursor.outer) {
                    Some(inner_list) => match inner_list.get(self.cursor.inner) {
                        Some(item) => {
                            self.cursor.inner += 1;
                            return Some(item.clone_ref(py));
                        }
                        None => {
                            self.cursor.outer += 1;
                            self.cursor.inner = 0;
                        }
                    },
                    None => {
                        self.done = true;
                        return None;
                    }
                }
            }
        }
    }
}

/// ref: `self.lists.iter().rev().flat_map(|x| x.iter().rev())`
pub struct SortedIterRev<T: InnerSorted> {
    owner: Py<T>,
    cursor: IterIdxs,
}

impl<T: InnerSorted> SortedIterRev<T> {
    pub fn new(owner: Py<T>) -> Self {
        Self {
            owner,
            cursor: IterIdxs { outer: 0, inner: 0 },
        }
    }
}

impl<T: InnerSorted> SortedIterator<T> for SortedIterRev<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        todo!()
    }
}

pub(crate) struct MinEqMaxIter<T: InnerSorted> {
    owner: Py<T>,
    bounds: IsliceBounds,
    current: usize,
}
impl<T: InnerSorted> MinEqMaxIter<T> {
    pub fn new(owner: Py<T>, bounds: IsliceBounds) -> Self {
        let current = bounds.min_idx.clone();
        Self {
            owner,
            bounds,
            current,
        }
    }
}
impl<T: InnerSorted> SortedIterator<T> for MinEqMaxIter<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let current = self.current;
        if current < self.bounds.max_idx {
            self.current += 1;
            Some(self.owner.get().get_lists()[self.bounds.min_pos][current].clone_ref(py))
        } else {
            None
        }
    }
}

pub(crate) struct MinEqMaxIterRev<T: InnerSorted> {
    owner: Py<T>,
    bounds: IsliceBounds,
    current: usize,
}
impl<T: InnerSorted> MinEqMaxIterRev<T> {
    pub fn new(owner: Py<T>, bounds: IsliceBounds) -> Self {
        let current = bounds.max_idx.clone();
        Self {
            owner,
            bounds,
            current,
        }
    }
}
impl<T: InnerSorted> SortedIterator<T> for MinEqMaxIterRev<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let current = self.current;
        if current > self.bounds.min_idx {
            self.current -= 1;
            Some(self.owner.get().get_lists()[self.bounds.min_pos][current].clone_ref(py))
        } else {
            None
        }
    }
}
enum NextMaxIterState {
    Min,
    Max,
    Done,
}
pub(crate) struct NextEqMaxIter<T: InnerSorted> {
    owner: Py<T>,
    bounds: IsliceBounds,
    state: NextMaxIterState,
}
impl<T: InnerSorted> NextEqMaxIter<T> {
    pub fn new(owner: Py<T>, bounds: IsliceBounds) -> Self {
        Self {
            owner,
            bounds,
            state: NextMaxIterState::Min,
        }
    }
}
impl<T: InnerSorted> SortedIterator<T> for NextEqMaxIter<T> {
    fn next(&mut self) -> Option<Py<PyAny>> {
        let lists = self.owner.get().get_lists();
        // NOTE: This code make no sense ATM. Placeholder.
        match self.state {
            NextMaxIterState::Min => (self.bounds.min_idx..lists[self.bounds.min_pos].len())
                .map(|x| lists[self.bounds.min_pos][x as usize])
                .next(),
            NextMaxIterState::Max => (0..self.bounds.max_idx)
                .map(|x| lists[self.bounds.max_pos][x as usize])
                .next(),
            NextMaxIterState::Done => None,
        }
    }
}
pub(crate) struct NextEqMaxIterRev<T: InnerSorted> {
    owner: Py<T>,
    bounds: IsliceBounds,
    state: NextMaxIterState,
}
impl<T: InnerSorted> NextEqMaxIterRev<T> {
    pub fn new(owner: Py<T>, bounds: IsliceBounds) -> Self {
        Self {
            owner,
            bounds,
            state: NextMaxIterState::Min,
        }
    }
}
impl<T: InnerSorted> SortedIterator<T> for NextEqMaxIterRev<T> {
    fn next(&mut self) -> Option<Py<PyAny>> {
        let lists = self.owner.get().get_lists();
        match self.state {
            NextMaxIterState::Max => (0..self.bounds.max_idx)
                .rev()
                .map(|x| lists[self.bounds.max_pos][x as usize])
                .next(),
            NextMaxIterState::Min => (self.bounds.min_idx..lists[self.bounds.min_pos].len())
                .rev()
                .map(|x| lists[self.bounds.min_pos][x as usize])
                .next(),
            NextMaxIterState::Done => None,
        }
    }
}
enum MinLtMaxIterState {
    Min,
    MinMax,
    Max,
    Done,
}
pub(crate) struct MinLtMaxIter<T: InnerSorted> {
    owner: Py<T>,
    bounds: IsliceBounds,
    current: usize,
    state: MinLtMaxIterState,
}
impl<T: InnerSorted> MinLtMaxIter<T> {
    pub fn new(owner: Py<T>, bounds: IsliceBounds) -> Self {
        Self {
            owner,
            bounds,
            current: bounds.min_idx.clone(),
            state: MinLtMaxIterState::Min,
        }
    }
}
impl<T: InnerSorted> SortedIterator<T> for MinLtMaxIter<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let lists = self.owner.get().get_lists();
        match self.state {
            MinLtMaxIterState::Done => None,
            MinLtMaxIterState::Min => (self.bounds.min_idx..lists[self.bounds.min_pos].len())
                .map(|x| lists[self.bounds.min_pos][x as usize])
                .next(),
            MinLtMaxIterState::MinMax => (self.bounds.min_pos + 1..self.bounds.max_pos)
                .flat_map(|x| lists[x].iter().map(|x| x.clone_ref(py)))
                .next(),
            MinLtMaxIterState::Max => (0..self.bounds.max_idx)
                .map(|x| lists[self.bounds.max_pos][x as usize])
                .next(),
        }
    }
}
pub(crate) struct MinLtMaxIterRev<T: InnerSorted> {
    owner: Py<T>,
    bounds: IsliceBounds,
    current: usize,
    state: MinLtMaxIterState,
}
impl<T: InnerSorted> MinLtMaxIterRev<T> {
    pub fn new(owner: Py<T>, bounds: IsliceBounds) -> Self {
        Self {
            owner,
            bounds,
            current: bounds.max_idx.clone(),
            state: MinLtMaxIterState::Max,
        }
    }
}
impl<T: InnerSorted> SortedIterator<T> for MinLtMaxIterRev<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let lists = self.owner.get().get_lists();
        match self.state {
            MinLtMaxIterState::Done => None,
            MinLtMaxIterState::Max => (0..self.bounds.max_idx)
                .rev()
                .map(|x| lists[self.bounds.max_pos][x as usize]),
            MinLtMaxIterState::MinMax => (self.bounds.min_pos + 1..self.bounds.max_pos)
                .rev()
                .flat_map(|x| lists[x].iter().rev().map(|y| y.clone_ref(py))),
            MinLtMaxIterState::Min => (self.bounds.min_idx..lists[self.bounds.min_pos].len())
                .rev()
                .map(|x| lists[self.bounds.min_pos][x as usize]),
        }
    }
}
