use crate::{
    abc,
    traits::{IntoPyochain, PyWrapper, PyoABC},
};
use either::Either;
use pyo3::{
    prelude::*,
    pyclass_init::PyClassInitializer,
    types::{PyInt, PyIterator, PyRange, PyRangeMethods, PySequence, PySlice, PyTuple},
    exceptions::PyTypeError,

};
use pyo3_ext::{prelude::*, pylibs};
use pyochain_macros::try_cast;
use tap::Pipe;

#[pyclass(module = "pyochain.core",frozen, sequence, extends=abc::PyoSequence)]
pub struct Range(pub Py<PyRange>);
impl Range {
    pub fn new(py: Python<'_>, start: isize, stop: isize) -> PyResult<Self> {
        PyRange::new(py, start, stop).map(Bound::unbind).map(Self)
    }
    pub fn new_with_step(py: Python<'_>, start: isize, stop: isize, step: isize) -> PyResult<Self> {
        PyRange::new_with_step(py, start, stop, step)
            .map(Bound::unbind)
            .map(Self)
    }
}
#[pymethods]
impl Range {
    #[pyo3(signature = (*args))]    
    #[new]
    fn py_new(
        _py: Python<'_>,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<PyClassInitializer<Self>> {

        let item_at = |idx: usize| unsafe {args.get_item_unchecked(idx).extract::<isize>()};
        
        let inner = match args.len(){
            1 => PyRange::new(item_at(0)?),
            2 => PyRange::new_with_step(item_at(0)?, item_at(1)?, 1),
            3 => PyRange::new_with_step(item_at(0)?, item_at(1)?, item_at(2)?),
            len => Err(PyTypeError::new_err(format!(
                "{} expected at most 3 arguments, got {len}", Self::NAME
            ))),
        }?;
        
        Ok(abc::PyoSequence::build_init().add_subclass(Self(inner)))
    }
    pub fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner_bind(py).iter_py()
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.inner_bind(py)
            .pipe(|x| unsafe { x.cast_unchecked::<PySequence>() })
            .len()
            .unwrap()
    }

    fn __getitem__<'py>(
        &self,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Either<Bound<'py, Self>, Bound<'py, PyAny>>> {
        let py = index.py();
        let range = self.inner_bind(py);
        try_cast! {
            match index {
                Case::PySlice(slice) => range
                    .get_item(slice)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyRange>() })?
                    .into_pyochain()
                    .map(Either::Left),
                object => range.get_item(object).map(Either::Right),
            }
        }
    }

    fn __repr__(slf: Bound<'_, Self>) -> String {
        let py = slf.py();
        let name = slf.get_type().name().unwrap();
        let inner = slf.get().inner_bind(py);

        let params = format!(
            "{}, {}, {}",
            inner.start().unwrap(),
            inner.stop().unwrap(),
            inner.step().unwrap()
        );
        format!("{}({})", name, params)
    }

    fn __eq__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).eq(value)
    }
    fn __hash__(slf: Bound<'_, Self>) -> PyResult<isize> {
        slf.get().inner_bind(slf.py()).hash()
    }
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(key.py()).contains(key)
    }
    pub fn __reversed__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner_bind(py).pipe_as_ref(pylibs::builtins::reversed)
    }
    #[pyo3(signature = (value, /))]
    fn count<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        let py = value.py();
        self.inner_bind(py).count(value)
    }
    fn index<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        let py = value.py();
        self.inner_bind(py).index(value)
    }
}
