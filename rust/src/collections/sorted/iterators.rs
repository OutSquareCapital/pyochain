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
    min_pos: usize,
    current: AtomicUsize,
    max_idx: usize,
}
impl MinEqMaxIter {
    pub fn new(
        owner: Bound<'_, InnerLists>,
        min_pos: usize,
        min_idx: usize,
        max_idx: usize,
    ) -> PyResult<Bound<'_, Self>> {
        let py = owner.py();
        let slf = abc::PyoIterator::build_init().add_subclass(Self {
            owner: owner.unbind(),
            min_pos,
            current: AtomicUsize::new(min_idx),
            max_idx,
        });
        Bound::new(py, slf)
    }
}
#[pymethods]
impl MinEqMaxIter {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let current = self.current.load(atomic::Ordering::Relaxed);
        if current < self.max_idx {
            self.current.store(current + 1, atomic::Ordering::Relaxed);
            Some(self.owner.get().get_lists()[self.min_pos][current].clone_ref(py))
        } else {
            None
        }
    }
}

#[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
pub(crate) struct MinEqMaxIterRev {
    owner: Py<InnerLists>,
    min_pos: usize,
    current: AtomicUsize,
    min_idx: usize,
}
impl MinEqMaxIterRev {
    pub fn new(
        owner: Bound<'_, InnerLists>,
        min_pos: usize,
        min_idx: usize,
        max_idx: usize,
    ) -> PyResult<Bound<'_, Self>> {
        let py = owner.py();
        let slf = abc::PyoIterator::build_init().add_subclass(Self {
            owner: owner.unbind(),
            min_pos,
            current: AtomicUsize::new(max_idx),
            min_idx: min_idx,
        });
        Bound::new(py, slf)
    }
}
#[pymethods]
impl MinEqMaxIterRev {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let current = self.current.load(atomic::Ordering::Relaxed);
        if current > self.min_idx {
            self.current.store(current - 1, atomic::Ordering::Relaxed);
            Some(self.owner.get().get_lists()[self.min_pos][current].clone_ref(py))
        } else {
            None
        }
    }
}
