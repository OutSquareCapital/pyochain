use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::{Bounds, ListDataGetters, Pos};

pub enum Dir {
    Fwd,
    Bwd,
}

pub struct ListDataIter<T: ListDataGetters> {
    data: Arc<Mutex<T>>,
    bounds: Bounds,
    direction: Dir,
}

impl<T: ListDataGetters> ListDataIter<T> {
    #[must_use]
    pub fn new(data: Arc<Mutex<T>>, bounds: Bounds, direction: Dir) -> Self {
        Self {
            data,
            bounds,
            direction,
        }
    }

    #[must_use]
    pub fn full(data: Arc<Mutex<T>>, direction: Dir) -> Self {
        let data_ref = data.lock().expect("poisoned");
        let last = data_ref.lists().len().saturating_sub(1);
        let bounds = Bounds::new(0, 0, last, data_ref.lists().last().map_or(0, Vec::len));
        drop(data_ref);
        Self::new(data, bounds, direction)
    }

    pub fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.bounds.min == self.bounds.max {
            None
        } else {
            let data = self.data.lock().expect("poisoned");
            match self.direction {
                Dir::Fwd => {
                    let item = data.lists()[self.bounds.min.pos][self.bounds.min.idx].clone_ref(py);
                    increment(&mut self.bounds.min, data.lists());
                    Some(item)
                }
                Dir::Bwd => {
                    decrement(&mut self.bounds.max, data.lists());
                    Some(data.lists()[self.bounds.max.pos][self.bounds.max.idx].clone_ref(py))
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
