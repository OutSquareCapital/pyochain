use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::{Bounds, ListDataGetters, Pos, traits::NestedVec};
struct ListDataIterInner<T: ListDataGetters> {
    data: Arc<Mutex<T>>,
    bounds: Bounds,
}

struct ListDataFullInner<T: ListDataGetters> {
    data: Arc<Mutex<T>>,
    pos: Pos,
}

pub struct Bounded<T: ListDataGetters>(ListDataIterInner<T>);
pub struct BoundedRev<T: ListDataGetters>(ListDataIterInner<T>);
pub struct Full<T: ListDataGetters>(ListDataFullInner<T>);
pub struct FullRev<T: ListDataGetters>(ListDataFullInner<T>);
pub trait ListDataIteratorMethods<T: ListDataGetters>: Sized {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>>;
}

impl<T: ListDataGetters> Bounded<T> {
    pub fn new(data: Arc<Mutex<T>>, bounds: Bounds) -> Self {
        Self(ListDataIterInner { data, bounds })
    }
}

impl<T: ListDataGetters> BoundedRev<T> {
    pub fn new(data: Arc<Mutex<T>>, bounds: Bounds) -> Self {
        Self(ListDataIterInner { data, bounds })
    }
}

impl<T: ListDataGetters> Full<T> {
    pub fn new(data: Arc<Mutex<T>>) -> Self {
        Self(ListDataFullInner {
            data,
            pos: Pos::default(),
        })
    }
}

impl<T: ListDataGetters> FullRev<T> {
    pub fn new(data: Arc<Mutex<T>>) -> Self {
        let data_ref = data.lock().expect("poisoned");
        let pos = Pos::new(
            data_ref.lists().len().saturating_sub(1),
            data_ref.lists().last().map_or(0, Vec::len),
        );
        drop(data_ref);
        Self(ListDataFullInner { data, pos })
    }
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for Full<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.0.data.lock().expect("poisoned");
        let bound = &mut self.0.pos;
        if bound.pos == data.lists().len() {
            None
        } else {
            let item = data.lists().iloc(bound).clone_ref(py);
            if bound.idx + 1 == data.lists()[bound.pos].len() {
                bound.pos += 1;
                bound.idx = 0;
            } else {
                bound.idx += 1;
            }
            Some(item)
        }
    }
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for FullRev<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.0.data.lock().expect("poisoned");
        let bound = &mut self.0.pos;
        if bound.pos == 0 && bound.idx == 0 {
            None
        } else {
            if bound.idx == 0 {
                bound.pos -= 1;
                bound.idx = data.lists()[bound.pos].len();
            }
            bound.idx -= 1;
            Some(data.lists().iloc(bound).clone_ref(py))
        }
    }
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for Bounded<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.0.bounds.min == self.0.bounds.max {
            None
        } else {
            let data = self.0.data.lock().expect("poisoned");
            let item = data.lists().iloc(&self.0.bounds.min).clone_ref(py);
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

impl<T: ListDataGetters> ListDataIteratorMethods<T> for BoundedRev<T> {
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
            Some(data.lists().iloc(&self.0.bounds.max).clone_ref(py))
        }
    }
}
