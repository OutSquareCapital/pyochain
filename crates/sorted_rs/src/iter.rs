use std::{
    ops::Deref,
    sync::{Arc, RwLock},
};

use pyo3::prelude::*;
use std_tools::prelude::*;

use crate::{Bounds, Loc, inner::InnerData, traits::NestedVec};
pub struct Bounded<T> {
    data: Arc<RwLock<T>>,
    bounds: Bounds,
}

pub struct Full<T> {
    data: Arc<RwLock<T>>,
    loc: Loc,
}

impl<T: Deref<Target = InnerData>> Bounded<T> {
    pub fn new(data: Arc<RwLock<T>>, bounds: Bounds) -> Self {
        Self { data, bounds }
    }
    pub fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.bounds.min == self.bounds.max {
            None
        } else {
            let data = self.data.read_or_inner();
            let item = data.values.loc(&self.bounds.min).clone_ref(py);
            let loc = &mut self.bounds.min;
            if loc.pos + 1 < data.values.len() && loc.idx + 1 >= data.values.loc_len(loc) {
                loc.pos += 1;
                loc.idx = 0;
            } else {
                loc.idx += 1;
            }
            Some(item)
        }
    }
    pub fn next_back(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.bounds.min == self.bounds.max {
            None
        } else {
            let data = self.data.read_or_inner();
            let loc = &mut self.bounds.max;

            if loc.idx > 0 {
                loc.idx -= 1;
            } else {
                loc.pos -= 1;
                loc.idx = data.values.loc_len(loc) - 1;
            }
            Some(data.values.loc(&self.bounds.max).clone_ref(py))
        }
    }
}
impl<T: Deref<Target = InnerData>> Full<T> {
    pub fn new(data: Arc<RwLock<T>>) -> Self {
        Self {
            data,
            loc: Loc::default(),
        }
    }
    pub fn new_rev(data: Arc<RwLock<T>>) -> Self {
        let data_ref = data.read_or_inner();
        let loc = Loc::new(
            data_ref.values.len().saturating_sub(1),
            data_ref.values.last().map_or(0, Vec::len),
        );
        drop(data_ref);
        Self { data, loc }
    }

    pub fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.data.read_or_inner();
        let loc = &mut self.loc;
        if loc.pos == data.values.len() {
            None
        } else {
            let item = data.values.loc(loc).clone_ref(py);
            if loc.idx + 1 == data.values.loc_len(loc) {
                loc.pos += 1;
                loc.idx = 0;
            } else {
                loc.idx += 1;
            }
            Some(item)
        }
    }
    pub fn next_back(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.data.read_or_inner();
        let loc = &mut self.loc;
        if loc.pos == 0 && loc.idx == 0 {
            None
        } else {
            if loc.idx == 0 {
                loc.pos -= 1;
                loc.idx = data.values.loc_len(loc);
            }
            loc.idx -= 1;
            Some(data.values.loc(loc).clone_ref(py))
        }
    }
}
