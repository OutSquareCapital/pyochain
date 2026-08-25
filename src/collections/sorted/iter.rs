use std::sync::Mutex;

use super::traits::BaseSortedList;
use crate::{
    abc,
    collections::{
        SortedKeyList, SortedList,
        sorted::bounds::{Bounds, Pos},
    },
};
use pyo3::prelude::*;

pub enum Dir {
    Fwd,
    Bwd,
}
pub(super) struct BoundedIter<T: BaseSortedList> {
    owner: Py<T>,
    bounds: Bounds,
    dir: Dir,
}

impl<T: BaseSortedList> BoundedIter<T> {
    pub fn new(owner: Py<T>, bounds: Bounds, dir: Dir) -> Self {
        Self { owner, bounds, dir }
    }

    pub fn full(owner: Py<T>, dir: Dir) -> Self {
        let data = owner.get().get_data();
        let last = data.lists.len().saturating_sub(1);
        let bounds = Bounds::new(0, 0, last, data.lists.last().map_or(0, Vec::len));
        drop(data);
        Self::new(owner, bounds, dir)
    }

    pub fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.bounds.min == self.bounds.max {
            None
        } else {
            let lists = &self.owner.get().get_data().lists;
            match self.dir {
                Dir::Fwd => {
                    let item = lists[self.bounds.min.pos][self.bounds.min.idx].clone_ref(py);
                    increment(&mut self.bounds.min, lists);
                    Some(item)
                }
                Dir::Bwd => {
                    decrement(&mut self.bounds.max, lists);
                    Some(lists[self.bounds.max.pos][self.bounds.max.idx].clone_ref(py))
                }
            }
        }
    }
}

fn increment(bound: &mut Pos, lists: &[Vec<Py<PyAny>>]) {
    if bound.pos + 1 < lists.len() && bound.idx + 1 >= lists[bound.pos].len() {
        bound.pos += 1;
        bound.idx = 0;
    } else {
        bound.idx += 1;
    }
}

fn decrement(bound: &mut Pos, lists: &[Vec<Py<PyAny>>]) {
    if bound.idx > 0 {
        bound.idx -= 1;
    } else {
        bound.pos -= 1;
        bound.idx = lists[bound.pos].len() - 1;
    }
}
#[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
pub struct SortedIter {
    inner: Mutex<BoundedIter<SortedList>>,
}
impl SortedIter {
    pub(super) fn new(inner: BoundedIter<SortedList>) -> Self {
        let inner = Mutex::new(inner);
        Self { inner }
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
    inner: Mutex<BoundedIter<SortedKeyList>>,
}
impl SortedIterKey {
    pub(super) fn new(inner: BoundedIter<SortedKeyList>) -> Self {
        let inner = Mutex::new(inner);
        Self { inner }
    }
}
#[pymethods]
impl SortedIterKey {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner.lock().expect("poisoned").next(py)
    }
}
