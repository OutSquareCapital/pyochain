use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::{Bounds, ListDataGetters};

pub struct ListDataIter<T: ListDataGetters> {
    data: Arc<Mutex<T>>,
    bounds: Bounds,
}

pub struct ListDataIterRev<T: ListDataGetters> {
    data: Arc<Mutex<T>>,
    bounds: Bounds,
}
pub trait ListDataIteratorMethods<T: ListDataGetters>: Sized {
    fn new(data: Arc<Mutex<T>>, bounds: Bounds) -> Self;
    fn full(data: Arc<Mutex<T>>) -> Self {
        let data_ref = data.lock().expect("poisoned");
        let last = data_ref.lists().len().saturating_sub(1);
        let bounds = Bounds::new(0, 0, last, data_ref.lists().last().map_or(0, Vec::len));
        drop(data_ref);
        Self::new(data, bounds)
    }
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>>;
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for ListDataIter<T> {
    fn new(data: Arc<Mutex<T>>, bounds: Bounds) -> Self {
        Self { data, bounds }
    }

    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.bounds.min == self.bounds.max {
            None
        } else {
            let data = self.data.lock().expect("poisoned");
            let item = data.lists()[self.bounds.min.pos][self.bounds.min.idx].clone_ref(py);
            let bound = &mut self.bounds.min;
            if bound.pos + 1 < data.lists().len() && bound.idx + 1 >= data.lists()[bound.pos].len()
            {
                bound.pos += 1;
                bound.idx = 0;
            } else {
                bound.idx += 1;
            }
            Some(item)
        }
    }
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for ListDataIterRev<T> {
    fn new(data: Arc<Mutex<T>>, bounds: Bounds) -> Self {
        Self { data, bounds }
    }

    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.bounds.min == self.bounds.max {
            None
        } else {
            let data = self.data.lock().expect("poisoned");
            let bound = &mut self.bounds.max;

            if bound.idx > 0 {
                bound.idx -= 1;
            } else {
                bound.pos -= 1;
                bound.idx = data.lists()[bound.pos].len() - 1;
            }
            Some(data.lists()[self.bounds.max.pos][self.bounds.max.idx].clone_ref(py))
        }
    }
}
