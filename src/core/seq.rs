use crate::{
    abc,
    display::get_repr,
    traits::{IntoPyochain, PyWrapper},
};
use either::Either;
use pyo3::{
    PyTypeInfo,
    prelude::*,
    types::{PyInt, PyIterator, PySequence, PySlice, PyTuple},
};
use pyo3_ext::prelude::*;
use pyochain_macros::try_cast;
use tap::Pipe;

#[pyclass(module = "pyochain.core",frozen, generic, sequence, extends=abc::PyoSequence)]
pub struct Seq(pub Py<PyTuple>);
#[pymethods]
impl Seq {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = Self::type_object(py).name()?;
        self.inner_bind(py)
            .pipe(get_repr)
            .map(|repr| format!("{name}({repr})"))
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner_bind(py).iter_py()
    }

    fn __len__(&self, py: Python) -> usize {
        self.inner_bind(py).len()
    }

    fn __getitem__<'py>(
        &self,
        index: Bound<'py, PyAny>,
    ) -> PyResult<Either<Bound<'py, Self>, Bound<'py, PyAny>>> {
        let tuple = self.inner_bind(index.py()).as_any();
        try_cast! {
            match index {
                Case::PySlice(slice) => tuple
                    .get_item(slice)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })?
                    .into_pyochain()
                    .map(Either::Left),
                object => tuple.get_item(object).map(Either::Right),
            }
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let py = other.py();
        let left = self.inner_bind(py);
        if let Ok(o) = other.cast_exact::<Self>() {
            left.eq(o.get().inner_bind(py)).unwrap()
        } else if let Ok(o) = other.cast_exact::<PyTuple>() {
            left.eq(o).unwrap()
        } else {
            false
        }
    }

    fn __hash__(&self, py: Python<'_>) -> isize {
        self.inner_bind(py).hash().unwrap()
    }
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(key.py()).contains(key)
    }
    fn __lt__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).lt(Self::extract_union(value)?)
    }
    fn __le__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).le(Self::extract_union(value)?)
    }
    fn __gt__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).gt(Self::extract_union(value)?)
    }
    fn __ge__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).ge(Self::extract_union(value)?)
    }
    fn __add__<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.concat(value)
    }
    fn __mul__<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.repeat(value)
    }
    fn __rmul__<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.repeat(value)
    }
    fn __repeat__<'py>(&self, py: Python<'py>, count: isize) -> PyResult<Bound<'py, Self>> {
        self.repeat(&PyInt::new(py, count))
    }
    fn __concat__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.concat(other)
    }
    fn __inplace_concat__<'py>(
        &self,
        other: &Bound<'py, PySequence>,
    ) -> PyResult<Bound<'py, PySequence>> {
        let py = other.py();
        let tup = Self::extract_union(other)?.as_sequence();
        self.inner_bind(py).as_sequence().in_place_concat(tup)
    }
    fn __inplace_repeat__<'py>(
        &self,
        py: Python<'py>,
        count: isize,
    ) -> PyResult<Bound<'py, PySequence>> {
        self.inner_bind(py)
            .as_sequence()
            .in_place_repeat(count.cast_unsigned())
    }
    #[pyo3(signature = (value, /))]
    fn count<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.inner_bind(value.py()).count(value)
    }
    #[pyo3(signature = (value, start = None, stop = None, /))]
    fn index<'py>(
        &self,
        value: &Bound<'py, PyAny>,
        start: Option<&Bound<'py, PyAny>>,
        stop: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        PySequenceExtMethods::index(self.inner_bind(value.py()), value, start, stop)
    }

    fn repeat<'py>(&self, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = n.py();
        self.inner_bind(py)
            .mul(n)
            .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
            .and_then(Bound::into_pyochain)
    }
    fn concat<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let other_seq = Self::extract_union(other)?.as_sequence();
        self.inner_bind(py)
            .as_sequence()
            .concat(other_seq)
            .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
            .and_then(Bound::into_pyochain)
    }
}
