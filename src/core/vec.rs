use crate::{abc, display::get_repr, traits::PyWrapper};

use either::Either;
use pyo3::{
    PyTypeInfo, ffi, intern,
    prelude::*,
    types::{PyDict, PyInt, PyIterator, PyList, PyNotImplemented, PySlice},
};
use pyo3_ext::{
    prelude::*,
    pylibs,
    types::{FromCmp, PyCmpOut},
};
use pyochain_macros::try_cast;
use tap::Pipe;
#[pyclass(module = "pyochain.core",frozen, generic, sequence, extends=abc::PyoMutableSequence, name="Vec")]
pub struct PyoVec(pub Py<PyList>);
#[pymethods]
impl PyoVec {
    fn __iter__<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner_bind(py).iter_py()
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(key.py()).contains(key)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = Self::type_object(py).name()?;
        self.inner_bind(py)
            .pipe_ref(get_repr)
            .map(|repr| format!("{name}({repr})"))
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.inner_bind(py).len()
    }

    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        let py = other.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match other {
                Case::PyList(list) => inner.eq(list).map(Either::Left),
                CaseExact::PyoVec(vec) => inner.eq(vec.get().inner_bind(py)).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }
    fn __reversed__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner_bind(py)
            .as_any()
            .pipe(pylibs::builtins::reversed)
    }

    fn __add__<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        Self::extract_union(value)?
            .as_sequence()
            .pipe(|x| self.inner_bind(py).as_sequence().concat(x))
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::try_into_py)
    }

    fn __iadd__(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        self.inner_bind(py).iadd(value)?;
        Ok(())
    }
    fn __inplace_concat__(&self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.__iadd__(other)
    }

    fn __concat__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.concat(other)
    }
    fn __mul__<'py>(&self, value: &Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        self.repeat(value)
    }
    fn __rmul__<'py>(&self, value: &Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        self.repeat(value)
    }
    fn __repeat__<'py>(&self, py: Python<'py>, count: isize) -> PyResult<Bound<'py, Self>> {
        self.repeat(&PyInt::new(py, count))
    }
    fn __inplace_repeat__(slf: Bound<'_, Self>, count: isize) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        Self::repeat_mut(slf, &PyInt::new(py, count))
    }
    fn __imul__(&self, py: Python<'_>, value: usize) -> PyResult<()> {
        self.inner_bind(py).as_sequence().in_place_repeat(value)?;
        Ok(())
    }

    fn __gt__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(value)?;
        self.inner_bind(py).gt(other)
    }

    fn __ge__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(value)?;
        self.inner_bind(py).ge(other)
    }

    fn __lt__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(value)?;
        self.inner_bind(py).lt(other)
    }

    fn __le__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(value)?;
        self.inner_bind(py).le(other)
    }

    fn __getitem__<'py>(
        &self,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Either<Bound<'py, Self>, Bound<'py, PyAny>>> {
        let list = self.inner_bind(index.py()).as_any();
        try_cast! {
            match index {
                Case::PySlice(slice) => list
                    .get_item(slice)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?
                    .try_into_py()
                    .map(Either::Left),
                object => list.get_item(object).map(Either::Right),
            }
        }
    }
    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(key.py()).as_any().set_item(key, value)
    }

    fn __delitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(key.py()).as_any().del_item(key)
    }

    #[pyo3(signature = (*, reverse=false))]
    fn sort(slf: Bound<'_, Self>, reverse: bool) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let list = slf.get().inner_bind(py);
        if reverse {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "reverse"), reverse)?;
            list.call_method(intern!(py, "sort"), (), Some(&kwargs))?;
        } else {
            list.sort()?;
        }
        Ok(slf)
    }
    #[pyo3(signature = (key, *, reverse=false))]
    fn sort_by<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        reverse: bool,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().inner_bind(slf.py()).sort_by(key, reverse)?;
        Ok(slf)
    }

    pub fn append(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).append(value)
    }

    pub fn extend(slf: &Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let inner = slf.get().inner_bind(py);
        let other = if iterable.is(slf) { inner } else { iterable };
        inner.extend(other)
    }

    fn repeat<'py>(&self, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = n.py();
        self.inner_bind(py)
            .mul(n)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::try_into_py)
    }
    fn repeat_mut<'py>(slf: Bound<'py, Self>, n: &Bound<'_, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = n.py();
        slf.get().inner_bind(py).imul(n)?;
        Ok(slf)
    }

    pub fn insert(&self, index: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let list = self.inner_bind(value.py()).as_ptr();
        match unsafe { ffi::PyList_Insert(list, index as ffi::Py_ssize_t, value.as_ptr()) } {
            -1 => Err(PyErr::fetch(py)),
            _ => Ok(()),
        }
    }

    pub fn clear(&self, py: Python<'_>) {
        self.inner_bind(py).clear();
    }

    fn reverse(&self, py: Python<'_>) -> PyResult<()> {
        self.inner_bind(py).reverse()
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(py).copy().and_then(Bound::try_into_py)
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

    fn concat<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        Self::extract_union(other)?
            .pipe(|other| {
                self.inner_bind(py)
                    .as_sequence()
                    .concat(other.as_sequence())
            })
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::try_into_py)
    }

    fn concat_mut<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let other = Self::extract_union(other)?.as_sequence();
        slf.get()
            .inner_bind(py)
            .as_sequence()
            .in_place_concat(other)
            .map(|_| slf)
    }
}
