use crate::{traits::ListsMethods, types::VecPy};
use pyo3::prelude::*;

pub trait Chunk<T>: FromIterator<T> {
    fn push_into(self, py: Python<'_>, list: &mut impl ListsMethods);
}

impl Chunk<Py<PyAny>> for VecPy {
    fn push_into(self, py: Python<'_>, list: &mut impl ListsMethods) {
        list.maxes.push(self.last().unwrap().clone_ref(py));
        list.keys_mut().push(self);
    }
}

impl Chunk<(Py<PyAny>, Py<PyAny>)> for (VecPy, VecPy) {
    fn push_into(self, py: Python<'_>, list: &mut impl ListsMethods) {
        let (keys, values) = self;
        keys.push_into(py, list);
        list.values.push(values);
    }
}
