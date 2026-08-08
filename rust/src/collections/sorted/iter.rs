use super::{
    InnerLists,
    traits::{InnerSorted, RustGetters},
};
use crate::{abc, iterators, traits::PyoABC};
use pyo3::prelude::*;
use std::sync::{
    Mutex,
    atomic::{self, AtomicUsize},
};
use tap::Pipe;
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
    pub fn into_iterator<T: InnerSorted>(
        self,
        slf: Bound<'_, T>,
        bounds: IsliceBounds,
    ) -> PyResult<Bound<'_, abc::PyoIterator>> {
        match self {
            Self::Empty => iterators::Iter::empty(slf.py())?.into_super(),
            Self::MinEqMax => MinEqMaxIter::new(slf, bounds)?.into_super(),
            Self::MinEqMaxRev => MinEqMaxIterRev::new(slf, bounds)?.into_super(),
            Self::NextEqMax => NextEqMaxIter::new(slf, bounds)?.into_super(),
            Self::NextEqMaxRev => NextEqMaxIterRev::new(slf, bounds)?.into_super(),
            Self::MinLtMax => MinLtMaxIter::new(slf, bounds)?.into_super(),
            Self::MinLtMaxRev => MinLtMaxIterRev::new(slf, bounds)?.into_super(),
        }
        .pipe(Ok)
    }
}

/// ref: `self.lists.iter().flatten()` for `__iter__`
/// ref: `self.lists.iter().rev().flat_map(|x| x.iter().rev())` for `__reversed__`
#[pyclass(module = "pyochain._iterators", frozen, extends = abc::PyoIterator)]
pub struct SortedIter {
    owner: Py<InnerLists>,
    cursor: Mutex<IterIdxs>,
}

impl SortedIter {
    pub fn new(owner: Bound<'_, InnerLists>) -> PyResult<Bound<'_, Self>> {
        let py = owner.py();
        let initializer = abc::PyoIterator::build_init().add_subclass(Self {
            owner: owner.unbind(),
            cursor: Mutex::new(IterIdxs { outer: 0, inner: 0 }),
        });
        Bound::new(py, initializer)
    }
}

#[pymethods]
impl SortedIter {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let mut idxs = self.cursor.lock().expect("Re-entrant error");
        let lists = self.owner.get().get_lists();
        while idxs.outer < lists.len() {
            let list = &lists[idxs.outer];
            if idxs.inner < list.len() {
                let item = list[idxs.inner].clone_ref(py);
                idxs.inner += 1;
                return Some(item);
            }
            idxs.outer += 1;
            idxs.inner = 0;
        }
        None
    }
}
#[pyclass(module = "pyochain._iterators", frozen, extends = abc::PyoIterator)]
pub struct SortedIterRev {
    owner: Py<InnerLists>,
    cursor: Mutex<IterIdxs>,
}

impl SortedIterRev {
    pub fn new(owner: Bound<'_, InnerLists>) -> PyResult<Bound<'_, Self>> {
        todo!()
    }
}

#[pymethods]
impl SortedIterRev {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        todo!()
    }
}

#[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
pub(crate) struct MinEqMaxIter {
    owner: Py<InnerLists>,
    bounds: IsliceBounds,
    current: AtomicUsize,
}
impl MinEqMaxIter {
    pub fn new(owner: Bound<'_, InnerLists>, bounds: IsliceBounds) -> PyResult<Bound<'_, Self>> {
        let py = owner.py();
        let current = AtomicUsize::new(bounds.min_idx.clone());
        let slf = abc::PyoIterator::build_init().add_subclass(Self {
            owner: owner.unbind(),
            bounds,
            current,
        });
        Bound::new(py, slf)
    }
}
#[pymethods]
impl MinEqMaxIter {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let current = self.current.load(atomic::Ordering::Relaxed);
        if current < self.bounds.max_idx {
            self.current.store(current + 1, atomic::Ordering::Relaxed);
            Some(self.owner.get().get_lists()[self.bounds.min_pos][current].clone_ref(py))
        } else {
            None
        }
    }
}

#[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
pub(crate) struct MinEqMaxIterRev {
    owner: Py<InnerLists>,
    bounds: IsliceBounds,
    current: AtomicUsize,
}
impl MinEqMaxIterRev {
    pub fn new(owner: Bound<'_, InnerLists>, bounds: IsliceBounds) -> PyResult<Bound<'_, Self>> {
        let py = owner.py();
        let current = AtomicUsize::new(bounds.max_idx.clone());
        let slf = abc::PyoIterator::build_init().add_subclass(Self {
            owner: owner.unbind(),
            bounds,
            current,
        });
        Bound::new(py, slf)
    }
}
#[pymethods]
impl MinEqMaxIterRev {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let current = self.current.load(atomic::Ordering::Relaxed);
        if current > self.bounds.min_idx {
            self.current.store(current - 1, atomic::Ordering::Relaxed);
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
#[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
pub(crate) struct NextEqMaxIter {
    owner: Py<InnerLists>,
    bounds: IsliceBounds,
    state: NextMaxIterState,
}
impl NextEqMaxIter {
    pub fn new(owner: Bound<'_, InnerLists>, bounds: IsliceBounds) -> PyResult<Bound<'_, Self>> {
        todo!()
    }
}
impl NextEqMaxIter {
    fn __next__(&self) -> Option<Py<PyAny>> {
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
#[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
pub(crate) struct NextEqMaxIterRev {
    owner: Py<InnerLists>,
    bounds: IsliceBounds,
    state: NextMaxIterState,
}
impl NextEqMaxIterRev {
    pub fn new(owner: Bound<'_, InnerLists>, bounds: IsliceBounds) -> PyResult<Bound<'_, Self>> {
        todo!()
    }
}
#[pymethods]
impl NextEqMaxIterRev {
    fn __next__(&self) -> Option<Py<PyAny>> {
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
#[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
pub(crate) struct MinLtMaxIter {
    owner: Py<InnerLists>,
    bounds: IsliceBounds,
    current: AtomicUsize,
    state: MinLtMaxIterState,
}
impl MinLtMaxIter {
    pub fn new(owner: Bound<'_, InnerLists>, bounds: IsliceBounds) -> PyResult<Bound<'_, Self>> {
        todo!()
    }
}

#[pymethods]
impl MinLtMaxIter {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
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
#[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
pub(crate) struct MinLtMaxIterRev {
    owner: Py<InnerLists>,
    bounds: IsliceBounds,
    current: AtomicUsize,
    state: MinLtMaxIterState,
}
impl MinLtMaxIterRev {
    pub fn new(owner: Bound<'_, InnerLists>, bounds: IsliceBounds) -> PyResult<Bound<'_, Self>> {
        todo!()
    }
}
#[pymethods]
impl MinLtMaxIterRev {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
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
