use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::{Bounds, ListDataGetters};
pub struct ListDataIterInner<T: ListDataGetters> {
    data: Arc<Mutex<T>>,
    bounds: Bounds,
}

pub struct ListDataIter<T: ListDataGetters>(ListDataIterInner<T>);
pub struct ListDataIterRev<T: ListDataGetters>(ListDataIterInner<T>);
pub trait ListDataIteratorMethods<T: ListDataGetters>: Sized {
    fn new(inner: ListDataIterInner<T>) -> Self;
    fn full(data: Arc<Mutex<T>>) -> Self {
        let data_ref = data.lock().expect("poisoned");
        let last = data_ref.lists().len().saturating_sub(1);
        let bounds = Bounds::new(0, 0, last, data_ref.lists().last().map_or(0, Vec::len));
        drop(data_ref);
        Self::new(ListDataIterInner { data, bounds })
    }
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>>;
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for ListDataIter<T> {
    fn new(inner: ListDataIterInner<T>) -> Self {
        Self(inner)
    }

    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.0.bounds.min == self.0.bounds.max {
            None
        } else {
            let data = self.0.data.lock().expect("poisoned");
            let item = data.lists()[self.0.bounds.min.pos][self.0.bounds.min.idx].clone_ref(py);
            let bound = &mut self.0.bounds.min;
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
    fn new(inner: ListDataIterInner<T>) -> Self {
        Self(inner)
    }

    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.0.bounds.min == self.0.bounds.max {
            None
        } else {
            let data = self.0.data.lock().expect("poisoned");
            let bound = &mut self.0.bounds.max;

            if bound.idx > 0 {
                bound.idx -= 1;
            } else {
                bound.pos -= 1;
                bound.idx = data.lists()[bound.pos].len() - 1;
            }
            Some(data.lists()[self.0.bounds.max.pos][self.0.bounds.max.idx].clone_ref(py))
        }
    }
}
