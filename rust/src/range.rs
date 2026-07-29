use crate::{
    abc,
    pyo3_ext::{prelude::*, pylibs},
    traits::PyoABC,
};
use pyo3::{
    prelude::*,
    pyclass_init::PyClassInitializer,
    types::{PyInt, PyIterator, PyRange, PyRangeMethods, PySequence},
};
use tap::Pipe;

#[pyclass(frozen, sequence, extends=abc::PyoSequence)]
pub struct Range {
    #[pyo3(get)]
    pub inner: Py<PyRange>,
}
impl Range {
    pub fn new(py: Python<'_>, start: isize, stop: isize) -> PyResult<Self> {
        PyRange::new(py, start, stop)
            .map(Bound::unbind)
            .map(|inner| Self { inner })
    }
    pub fn new_with_step(py: Python<'_>, start: isize, stop: isize, step: isize) -> PyResult<Self> {
        PyRange::new_with_step(py, start, stop, step)
            .map(Bound::unbind)
            .map(|inner| Self { inner })
    }
}
#[pymethods]
impl Range {
    #[pyo3(signature = (start, stop, step = 1))]
    #[new]
    fn py_new(
        py: Python<'_>,
        start: isize,
        stop: isize,
        step: isize,
    ) -> PyResult<PyClassInitializer<Self>> {
        PyRange::new_with_step(py, start, stop, step)
            .map(Bound::unbind)
            .map(|inner| abc::PyoSequence::build_init().add_subclass(Self { inner }))
    }
    pub fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner.bind(py).try_iter().unwrap()
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.inner
            .bind(py)
            .pipe(|x| unsafe { x.cast_unchecked::<PySequence>() })
            .len()
            .unwrap()
    }

    fn __getitem__<'py>(&self, index: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = index.py();
        self.inner.bind(py).get_item(index)
    }

    fn __repr__(slf: Bound<'_, Self>) -> String {
        let py = slf.py();
        let name = slf.get_type().name().unwrap();
        let inner = slf.get().inner.bind(py);

        let params = format!(
            "{}, {}, {}",
            inner.start().unwrap(),
            inner.stop().unwrap(),
            inner.step().unwrap()
        );
        format!("{}({})", name, params)
    }

    fn __eq__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).eq(value)
    }
    fn __hash__(slf: Bound<'_, Self>) -> PyResult<isize> {
        slf.get().inner.bind(slf.py()).hash()
    }
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(key.py()).contains(key)
    }
    pub fn __reversed__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner.bind(py).pipe_as_ref(pylibs::builtins::reversed)
    }
    #[pyo3(signature = (value, /))]
    fn count<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        let py = value.py();
        self.inner.bind(py).count(value)
    }
    fn index<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        let py = value.py();
        self.inner.bind(py).index(value)
    }
}
