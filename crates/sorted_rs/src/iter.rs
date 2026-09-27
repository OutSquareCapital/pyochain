use crate::{Bounds, bounds::AtomicBounds, inner::InnerData};
use derive_more::Constructor;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::{ops::Deref, sync::Arc};
#[derive(Constructor)]
pub struct Bounded<T> {
    data: Arc<RwLock<T>>,
    bounds: AtomicBounds,
    reversed: bool,
}

impl<T: Deref<Target = InnerData>> Bounded<T> {
    pub fn full(data: Arc<RwLock<T>>, reversed: bool) -> Self {
        let bounds = Bounds::from_full_iter(&data.read());
        Self::new(data, bounds.into(), reversed)
    }

    #[inline]
    pub fn next(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        let data = self.data.read();
        let b = &self.bounds;
        let (min_pos, min_idx) = b.min.load();
        let (max_pos, max_idx) = b.max.load();
        if min_pos == max_pos && min_idx == max_idx {
            None
        } else if self.reversed {
            let (new_pos, new_idx) = if max_idx > 0 {
                (max_pos, max_idx - 1)
            } else {
                let p = max_pos - 1;
                (p, data.values[p].len() - 1)
            };
            let item = data.values[new_pos][new_idx].clone_ref(py);
            b.max.store(new_pos, new_idx);
            Some(item)
        } else {
            let item = data.values[min_pos][min_idx].clone_ref(py);
            let v = &data.values[min_pos];
            let (new_pos, new_idx) = if min_pos + 1 < data.values.len() && min_idx + 1 >= v.len() {
                (min_pos + 1, 0)
            } else {
                (min_pos, min_idx + 1)
            };
            b.min.store(new_pos, new_idx);
            Some(item)
        }
    }
}
