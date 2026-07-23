use crate::seq::PyoVec;
use pyo3::prelude::*;
const DEFAULT_LOAD_FACTOR: usize = 1000;
#[pyclass(generic)]
pub struct InnerLists {
    #[pyo3(get, set)]
    lists: Py<PyoVec>,
    #[pyo3(get, set)]
    maxes: Py<PyoVec>,
    #[pyo3(get, set)]
    idx: Py<PyoVec>,
    #[pyo3(get, set)]
    len: usize,
    #[pyo3(get, set)]
    load: usize,
    #[pyo3(get, set)]
    offset: usize,
}
#[pymethods]
impl InnerLists {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            lists: PyoVec::new_bound(py)?.unbind(),
            maxes: PyoVec::new_bound(py)?.unbind(),
            idx: PyoVec::new_bound(py)?.unbind(),
            len: 0,
            load: DEFAULT_LOAD_FACTOR,
            offset: 0,
        })
    }

    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        self.len = 0;
        self.lists.get().clear(py)?;
        self.maxes.get().clear(py)?;
        self.idx.get().clear(py)?;
        self.offset = 0;
        Ok(())
    }
}

#[pyclass(generic)]
pub struct InnerKeyLists {
    #[pyo3(get, set)]
    key: Py<PyAny>,
    #[pyo3(get, set)]
    keys: Py<PyoVec>,
    #[pyo3(get, set)]
    lists: Py<PyoVec>,
    #[pyo3(get, set)]
    maxes: Py<PyoVec>,
    #[pyo3(get, set)]
    idx: Py<PyoVec>,
    #[pyo3(get, set)]
    len: usize,
    #[pyo3(get, set)]
    load: usize,
    #[pyo3(get, set)]
    offset: usize,
}
#[pymethods]
impl InnerKeyLists {
    #[new]
    fn new(key: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = key.py();
        Ok(Self {
            key: key.unbind(),
            keys: PyoVec::new_bound(py)?.unbind(),
            lists: PyoVec::new_bound(py)?.unbind(),
            maxes: PyoVec::new_bound(py)?.unbind(),
            idx: PyoVec::new_bound(py)?.unbind(),
            len: 0,
            load: DEFAULT_LOAD_FACTOR,
            offset: 0,
        })
    }

    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        self.len = 0;
        self.lists.get().clear(py)?;
        self.keys.get().clear(py)?;
        self.maxes.get().clear(py)?;
        self.idx.get().clear(py)?;
        Ok(())
    }
}
