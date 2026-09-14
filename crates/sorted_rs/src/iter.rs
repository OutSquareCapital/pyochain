use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use std_tools::prelude::*;

use crate::{Bounds, ListDataGetters, Loc, traits::NestedVec};
struct ListDataIterInner<T: ListDataGetters> {
    data: Arc<Mutex<T>>,
    bounds: Bounds,
}

struct ListDataFullInner<T: ListDataGetters> {
    data: Arc<Mutex<T>>,
    loc: Loc,
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
            loc: Loc::default(),
        })
    }
}

impl<T: ListDataGetters> FullRev<T> {
    pub fn new(data: Arc<Mutex<T>>) -> Self {
        let data_ref = data.try_into_inner();
        let loc = Loc::new(
            data_ref.values().len().saturating_sub(1),
            data_ref.values().last().map_or(0, Vec::len),
        );
        drop(data_ref);
        Self(ListDataFullInner { data, loc })
    }
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for Full<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.0.data.try_into_inner();
        let loc = &mut self.0.loc;
        if loc.pos == data.values().len() {
            None
        } else {
            let item = data.values().loc(loc).clone_ref(py);
            if loc.idx + 1 == data.values().loc_len(loc) {
                loc.pos += 1;
                loc.idx = 0;
            } else {
                loc.idx += 1;
            }
            Some(item)
        }
    }
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for FullRev<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.0.data.try_into_inner();
        let loc = &mut self.0.loc;
        if loc.pos == 0 && loc.idx == 0 {
            None
        } else {
            if loc.idx == 0 {
                loc.pos -= 1;
                loc.idx = data.values().loc_len(loc);
            }
            loc.idx -= 1;
            Some(data.values().loc(loc).clone_ref(py))
        }
    }
}

impl<T: ListDataGetters> ListDataIteratorMethods<T> for Bounded<T> {
    fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.0.bounds.min == self.0.bounds.max {
            None
        } else {
            let data = self.0.data.try_into_inner();
            let item = data.values().loc(&self.0.bounds.min).clone_ref(py);
            let loc = &mut self.0.bounds.min;
            if loc.pos + 1 < data.values().len() && loc.idx + 1 >= data.values().loc_len(loc) {
                loc.pos += 1;
                loc.idx = 0;
            } else {
                loc.idx += 1;
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
            let data = self.0.data.try_into_inner();
            let loc = &mut self.0.bounds.max;

            if loc.idx > 0 {
                loc.idx -= 1;
            } else {
                loc.pos -= 1;
                loc.idx = data.values().loc_len(loc) - 1;
            }
            Some(data.values().loc(&self.0.bounds.max).clone_ref(py))
        }
    }
}
