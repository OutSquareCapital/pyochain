use super::{InnerKeyLists, InnerLists, traits::RustGetters};
use crate::{abc, traits::PyoABC};
use pyo3::prelude::*;
use std::sync::{
    Mutex, MutexGuard,
    atomic::{self, AtomicUsize},
};
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
pub trait SortedIterator: abc::traits::ImplPyoIterator {
    fn get_lists(&self) -> MutexGuard<'_, Vec<Vec<Py<PyAny>>>>;
    fn get_cursor(&self) -> MutexGuard<'_, IterIdxs>;
    #[inline]
    fn next(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let mut idxs = self.get_cursor();
        let lists = self.get_lists();
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

/// ref: `self.lists.iter().flatten()`
macro_rules! impl_sorted_iter {
    ($iter_name:ident, $owner:ty) => {
        impl SortedIterator for $iter_name {
            fn get_lists(&self) -> MutexGuard<'_, Vec<Vec<Py<PyAny>>>> {
                self.owner.get().get_lists()
            }
            fn get_cursor(&self) -> MutexGuard<'_, IterIdxs> {
                self.cursor.lock().expect("Failed to lock cursor mutex")
            }
        }
        #[pyclass(module = "pyochain._iterators", frozen, extends = abc::PyoIterator)]
        pub struct $iter_name {
            owner: Py<$owner>,
            cursor: Mutex<IterIdxs>,
        }

        impl $iter_name {
            pub fn new(py: Python<'_>, owner: Py<$owner>) -> PyResult<Bound<'_, Self>> {
                let initializer = abc::PyoIterator::build_init().add_subclass(Self {
                    owner,
                    cursor: Mutex::new(IterIdxs { outer: 0, inner: 0 }),
                });
                Bound::new(py, initializer)
            }
        }

        #[pymethods]
        impl $iter_name {
            fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                self.next(py)
            }
        }
    };
}

impl_sorted_iter!(SortedListIter, InnerLists);
impl_sorted_iter!(SortedKeyListIter, InnerKeyLists);

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
    pub fn new(min_pos: usize, max_pos: usize, reverse: bool) -> Self {
        let next_pos = min_pos + 1;
        if min_pos > max_pos {
            return Self::Empty;
        }

        if min_pos == max_pos {
            if reverse {
                return Self::MinEqMaxRev;
            }
            return Self::MinEqMax;
        }

        if next_pos == max_pos {
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
        (0..max_idx)
            .rev()
            .map(|x| lists[max_pos][x as usize])
            .chain(
                (min_idx..lists[min_pos].len() as isize)
                    .rev()
                    .map(|x| lists[min_pos][x as usize]),
            )
    }
}
