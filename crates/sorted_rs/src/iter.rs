use crate::{Bounds, Loc, inner::InnerData, traits::NestedVec};
use derive_more::Constructor;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::{ops::Deref, sync::Arc};
#[derive(Constructor)]
pub struct Bounded<T> {
    data: Arc<RwLock<T>>,
    bounds: Bounds,
}
impl<T: Deref<Target = InnerData>> Bounded<T> {
    pub fn full(data: Arc<RwLock<T>>) -> Self {
        let data_ref = data.read();
        let max = data_ref.values.last().map_or(Loc::default(), |v| {
            Loc::new(data_ref.values.len() - 1, v.len())
        });
        drop(data_ref);
        Self {
            data,
            bounds: Bounds::new(Loc::default(), max),
        }
    }
    #[inline]
    pub fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.data.read();
        let loc = &mut self.bounds.min;
        if loc == &self.bounds.max {
            None
        } else {
            let v = &data.values[loc.pos];
            let item = v[loc.idx].clone_ref(py);
            if loc.pos + 1 < data.values.len() && loc.idx + 1 >= v.len() {
                loc.pos += 1;
                loc.idx = 0;
            } else {
                loc.idx += 1;
            }
            Some(item)
        }
    }
    #[inline]
    pub fn next_back(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.data.read();
        let loc = &mut self.bounds.max;
        if &self.bounds.min == loc {
            None
        } else {
            if loc.idx > 0 {
                loc.idx -= 1;
            } else {
                loc.pos -= 1;
                loc.idx = data.values.loc_len(loc) - 1;
            }
            Some(data.values.loc(loc).clone_ref(py))
        }
    }
}
