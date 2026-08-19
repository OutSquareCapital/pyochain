use crate::{
    abc,
    display::get_repr,
    traits::{IntoPyochain, PyWrapper, PyoABC},
};

use either::Either;
use pyo3::{
    ffi, intern,
    prelude::*,
    pyclass_init::PyClassInitializer,
    types::{PyDict, PyInt, PyIterator, PyList, PyNotImplemented, PySequence, PySlice, PyTuple},
};
use pyo3_ext::{
    prelude::*,
    pylibs,
    types::{FromCmp, PyCmpOut, PyIterable},
};
use pyochain_macros::{try_cast, try_cast_into};
use tap::Pipe;
#[pyclass(module = "pyochain.core",frozen, generic, sequence, extends=abc::PyoMutableSequence, name="Vec")]
pub struct PyoVec(pub Py<PyList>);
#[pymethods]
impl PyoVec {
    #[pyo3(signature = (data = None, *more))]
    #[new]
    fn new(
        data: Option<Bound<'_, PyAny>>,
        more: Bound<'_, PyTuple>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = more.py();
        let list = try_cast_into! {
            match (data, more.is_empty()) {
                (None, _) => more.to_list(),
                (Some(CaseExact::Self(inner)), true) => {
                    inner.get().into_inner_bound(py).as_sequence().to_list()?
                }
                (Some(Case::PySequence(sequence)), true) => sequence.to_list()?,
                (Some(Case::PyIterable(iterable)), true) => {
                    iterable.try_iter()?.try_collect_bound(py)?
                }
                (Some(any), true) => PyList::new(py, [any])?,
                (Some(CaseExact::Self(inner)), false) => inner
                    .get()
                    .into_inner_bound(py)
                    .as_sequence()
                    .concat(&more.as_sequence())?
                    .pipe(|x| unsafe { x.cast_into_unchecked::<PyList>() }),
                (Some(Case::PySequence(sequence)), false) => sequence
                    .to_list()?
                    .as_sequence()
                    .in_place_concat(&more.as_sequence())?
                    .pipe(|x| unsafe { x.cast_into_unchecked::<PyList>() }),
                (Some(Case::PyIterable(iterable)), false) => iterable
                    .try_iter()?
                    .into_iter()
                    .chain(more.into_iter().map(Ok))
                    .try_collect_bound(py)?,
                (Some(any), false) => std::iter::once(any)
                    .chain(more.into_iter())
                    .collect_bound(py)?,
            }
        };
        abc::PyoMutableSequence::build_init()
            .add_subclass(Self(list.unbind()))
            .pipe(Ok)
    }
    #[staticmethod]
    fn from_ref<'py>(data: Bound<'py, PyList>) -> PyResult<Bound<'py, Self>> {
        data.into_pyochain()
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner_bind(slf.py()).iter_py()
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(key.py()).contains(key)
    }
    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;

        slf.get()
            .into_inner_bound(py)
            .pipe(get_repr)
            .map(|repr| format!("{}({})", name, repr))
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        slf.get().inner_bind(slf.py()).len()
    }

    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        let py = other.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match other {
                Case::PyList(list) => inner.eq(&list).map(Either::Left),
                CaseExact::PyoVec(vec) => inner.eq(&vec.get().inner_bind(py)).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }
    fn __reversed__<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyIterator> {
        let py = slf.py();
        slf.get()
            .inner_bind(py)
            .pipe_as_ref(pylibs::builtins::reversed)
    }

    fn __add__<'py>(slf: Bound<'py, Self>, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        Self::extract_union(&value)?
            .as_sequence()
            .pipe(|x| slf.get().inner_bind(py).as_sequence().concat(x))
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::into_pyochain)
    }

    fn __iadd__<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<()> {
        let py = value.py();
        self.inner_bind(py).iadd(value)?;
        Ok(())
    }
    fn __inplace_concat__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<()> {
        self.__iadd__(other)
    }

    fn __concat__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.concat(other)
    }
    fn __mul__<'py>(slf: Bound<'_, Self>, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&value)
    }
    fn __rmul__<'py>(
        slf: Bound<'py, Self>,
        value: Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&value)
    }
    fn __repeat__<'py>(slf: Bound<'py, Self>, count: isize) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&PyInt::new(slf.py(), count))
    }
    fn __inplace_repeat__<'py>(slf: Bound<'py, Self>, count: isize) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        Self::repeat_mut(slf, &PyInt::new(py, count))
    }
    fn __imul__(slf: Bound<'_, Self>, value: usize) -> PyResult<()> {
        slf.get()
            .inner_bind(slf.py())
            .as_sequence()
            .in_place_repeat(value)?;
        Ok(())
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(&value)?;
        self.inner_bind(py).gt(other)
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(&value)?;
        self.inner_bind(py).ge(other)
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(&value)?;
        self.inner_bind(py).lt(other)
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(&value)?;
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
                    .into_pyochain()
                    .map(Either::Left),
                object => list.get_item(object).map(Either::Right),
            }
        }
    }
    fn __setitem__(
        slf: Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        slf.get().inner_bind(key.py()).as_any().set_item(key, value)
    }

    fn __delitem__(slf: Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.get().inner_bind(key.py()).as_any().del_item(key)
    }

    #[pyo3(signature = (*, reverse=false))]
    fn sort(slf: Bound<'_, Self>, reverse: bool) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "reverse"), reverse)?;
        slf.get()
            .inner_bind(py)
            .call_method(intern!(py, "sort"), (), Some(&kwargs))?;
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

    pub fn extend(slf: Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let inner = slf.get().inner_bind(py);
        let other = if iterable.is(&slf) { inner } else { iterable };
        inner.extend(other)
    }

    fn repeat<'py>(&self, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = n.py();
        self.inner_bind(py)
            .mul(n)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::into_pyochain)
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

    pub fn clear(&self, py: Python<'_>) -> () {
        self.inner_bind(py).clear()
    }

    fn reverse(slf: Bound<'_, Self>) -> PyResult<()> {
        slf.get().inner_bind(slf.py()).reverse()
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get()
            .inner_bind(slf.py())
            .call_method0(intern!(slf.py(), "copy"))
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::into_pyochain)
    }
    #[pyo3(signature = (value, /))]
    fn count<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.inner_bind(value.py()).count(&value)
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
        Self::extract_union(&other)?
            .pipe(|other| {
                self.inner_bind(py)
                    .as_sequence()
                    .concat(other.as_sequence())
            })
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::into_pyochain)
    }

    fn concat_mut<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let other = Self::extract_union(&other)?.as_sequence();
        slf.get()
            .inner_bind(py)
            .as_sequence()
            .in_place_concat(other)
            .map(|_| slf)
    }
}
