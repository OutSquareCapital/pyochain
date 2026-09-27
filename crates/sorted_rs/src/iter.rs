use crate::{Bounds, inner::InnerData, traits::NestedVec};
use derive_more::Constructor;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::{ops::Deref, sync::Arc};
#[derive(Constructor)]
pub struct Bounded<T> {
    data: Arc<RwLock<T>>,
    bounds: Bounds,
    reversed: bool,
}
impl<T: Deref<Target = InnerData>> Bounded<T> {
    pub fn full(data: Arc<RwLock<T>>, reversed: bool) -> Self {
        let bounds = Bounds::from_full_iter(&data.read());
        Self::new(data, bounds, reversed)
    }
    #[inline]
    pub fn next(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.data.read();
        let max = &mut self.bounds.max;
        let min = &mut self.bounds.min;
        if max == min {
            None
        } else if self.reversed {
            if max.idx > 0 {
                max.idx -= 1;
            } else {
                max.pos -= 1;
                max.idx = data.values.loc_len(max) - 1;
            }
            Some(data.values.loc(max).clone_ref(py))
        } else {
            let v = &data.values[min.pos];
            let item = v[min.idx].clone_ref(py);
            if min.pos + 1 < data.values.len() && min.idx + 1 >= v.len() {
                min.pos += 1;
                min.idx = 0;
            } else {
                min.idx += 1;
            }
            Some(item)
        }
    }
}
