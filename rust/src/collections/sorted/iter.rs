use std::sync::Mutex;

use super::traits::InnerSorted;
use crate::{
    abc,
    collections::{InnerKeyLists, InnerLists},
    traits::PyoABC,
};
use pyo3::prelude::*;
use tap::Pipe;

#[derive(PartialEq, Eq)]
struct Pos {
    pos: usize,
    idx: usize,
}
impl Pos {
    fn increment(&mut self, lists: &[Vec<Py<PyAny>>]) -> () {
        if self.pos + 1 < lists.len() && self.idx + 1 >= lists[self.pos].len() {
            self.pos += 1;
            self.idx = 0;
        } else {
            self.idx += 1;
        }
    }

    fn decrement(&mut self, lists: &[Vec<Py<PyAny>>]) -> () {
        if self.idx > 0 {
            self.idx -= 1;
        } else {
            self.pos -= 1;
            self.idx = lists[self.pos].len() - 1;
        }
    }
}

pub enum Dir {
    Fwd,
    Bwd,
}
pub(super) struct IsliceBounds {
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
    #[inline]
    pub fn from_irange_spec(
        min_pos: usize,
        min_idx: usize,
        max_pos: usize,
        max_idx: usize,
    ) -> Option<Self> {
        if min_pos > max_pos || (min_pos == max_pos && min_idx >= max_idx) {
            None
        } else {
            Some(Self::new(min_pos, min_idx, max_pos, max_idx))
        }
    }
}

pub(super) struct BoundedIter<T: InnerSorted> {
    owner: Py<T>,
    start: Pos,
    end: Pos,
    dir: Dir,
}

impl<T: InnerSorted> BoundedIter<T> {
    pub fn new(owner: Py<T>, bounds: IsliceBounds, dir: Dir) -> Self {
        Self {
            owner,
            start: Pos {
                pos: bounds.min_pos,
                idx: bounds.min_idx,
            },
            end: Pos {
                pos: bounds.max_pos,
                idx: bounds.max_idx,
            },
            dir,
        }
    }

    pub fn full(owner: Py<T>, dir: Dir) -> Self {
        let lists = owner.get().get_lists();
        let last = lists.len().saturating_sub(1);
        let bounds = IsliceBounds::new(0, 0, last, lists.last().map_or(0, |x| x.len()));
        drop(lists);
        Self::new(owner, bounds, dir)
    }

    pub fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.start == self.end {
            None
        } else {
            let lists = self.owner.get().get_lists();
            match self.dir {
                Dir::Fwd => {
                    let item = lists[self.start.pos][self.start.idx].clone_ref(py);
                    self.start.increment(&lists);
                    Some(item)
                }
                Dir::Bwd => {
                    self.end.decrement(&lists);
                    Some(lists[self.end.pos][self.end.idx].clone_ref(py))
                }
            }
        }
    }
}

#[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
pub struct SortedIter {
    inner: Mutex<BoundedIter<InnerLists>>,
}
impl SortedIter {
    pub(super) fn build<'py>(
        py: Python<'py>,
        inner: BoundedIter<InnerLists>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        abc::PyoIterator::build_init()
            .add_subclass(Self {
                inner: Mutex::new(inner),
            })
            .pipe(|x| Bound::new(py, x))?
            .into_super()
            .pipe(Ok)
    }
}
#[pymethods]
impl SortedIter {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner.lock().expect("poisoned").next(py)
    }
}

#[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
pub struct SortedIterKey {
    inner: Mutex<BoundedIter<InnerKeyLists>>,
}
impl SortedIterKey {
    pub(super) fn build<'py>(
        py: Python<'py>,
        inner: BoundedIter<InnerKeyLists>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        abc::PyoIterator::build_init()
            .add_subclass(Self {
                inner: Mutex::new(inner),
            })
            .pipe(|x| Bound::new(py, x))?
            .into_super()
            .pipe(Ok)
    }
}
#[pymethods]
impl SortedIterKey {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner.lock().expect("poisoned").next(py)
    }
}
