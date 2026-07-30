use crate::{
    abc,
    display::get_repr,
    pyo3_ext::{prelude::*, types::PyCmpOut},
    traits::{IntoPyochain, PyoABC},
};
use either::Either;
use pyo3::{
    BoundObject, PyTypeInfo, intern,
    prelude::*,
    pyclass_init::PyClassInitializer,
    types::{PyBool, PyFrozenSet, PyIterator, PyNotImplemented, PySet, PyTuple},
};
use pyochain_macros::try_cast;
use tap::Pipe;

#[pyclass(frozen, generic, extends=abc::PyoSet)]
pub struct Set {
    #[pyo3(get)]
    pub inner: Py<PyFrozenSet>,
}
#[pymethods]
impl Set {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        data.pipe(|x| PyFrozenSet::type_object(py).call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
            .map(Bound::unbind)
            .map(|inner| abc::PyoSet::build_init().add_subclass(Self { inner }))
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;
        slf.get()
            .inner
            .bind(py)
            .pipe(|x| PyTuple::new(py, x))
            .and_then(get_repr)
            .map(|repr| format!("{}({})", name, repr))
    }

    fn __contains__(&self, item: Bound<'_, PyAny>) -> PyResult<bool> {
        return self.inner.bind(item.py()).contains(item);
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        return slf.get().inner.bind(slf.py()).try_iter().unwrap();
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        return slf.get().inner.bind(slf.py()).len();
    }

    fn __and__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .bitand(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
            .and_then(Bound::into_pyochain)
    }
    fn __or__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .bitor(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
            .and_then(Bound::into_pyochain)
    }

    fn __sub__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .sub(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
            .and_then(Bound::into_pyochain)
    }

    fn __rsub__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        value
            .sub(self.inner.bind(value.py()))
            .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
            .and_then(Bound::into_pyochain)
    }
    fn __xor__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .bitxor(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
            .and_then(Bound::into_pyochain)
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).le(value)
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).lt(value)
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).ge(value)
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).gt(value)
    }

    fn __eq__<'py>(&self, value: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        self.inner
            .bind(value.py())
            .as_any()
            .pipe_ref(|x| set_eq(x, value))
    }

    fn __hash__(slf: Bound<'_, Self>) -> PyResult<isize> {
        slf.get().inner.bind(slf.py()).hash()
    }
    fn __rand__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(value)
    }

    fn __ror__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__or__(value)
    }
    fn __rxor__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__xor__(value)
    }
    fn isdisjoint<'py>(&self, s: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner.bind(s.py()).isdisjoint(s)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner.bind(other.py()).issubset(other)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner.bind(other.py()).issuperset(other)
    }
    #[pyo3(signature = (*others))]
    fn intersection<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(others.py())
            .intersection(others)
            .and_then(Bound::into_pyochain)
    }

    #[pyo3(signature = (*others))]
    fn union<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(others.py())
            .union(others)
            .and_then(Bound::into_pyochain)
    }

    #[pyo3(signature = (*others))]
    fn difference<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(others.py())
            .difference(others)
            .and_then(Bound::into_pyochain)
    }

    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(other.py())
            .symmetric_difference(other)
            .and_then(Bound::into_pyochain)
    }
}
#[pyclass(generic, frozen, extends=abc::PyoMutableSet)]
pub struct SetMut {
    #[pyo3(get)]
    pub inner: Py<PySet>,
}
#[pymethods]
impl SetMut {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        data.pipe(|x| PySet::type_object(py).call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .map(Bound::unbind)
            .map(|inner| abc::PyoMutableSet::build_init().add_subclass(SetMut { inner }))
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner.bind(slf.py()).try_iter().unwrap()
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        slf.get().inner.bind(slf.py()).len()
    }

    fn __contains__(&self, item: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(item.py()).contains(item)
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;
        slf.get()
            .inner
            .bind(py)
            .pipe(|x| PyTuple::new(py, x))
            .and_then(get_repr)
            .map(|repr| format!("{}({})", name, repr))
    }

    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        self.inner
            .bind(other.py())
            .as_any()
            .pipe_ref(|x| set_eq(x, other))
    }

    fn __and__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .bitand(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }

    fn __or__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .bitor(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }
    /// NOTE: We need to use `call_method1` for in-place operators here because Pyo3 doesn't allow returning something else than `PyResult<()>`.\
    /// And if we use `PySet::__ior__`, it will return `NotImplemented` on object who are NOT subclasses of `set` or `frozenset`.\
    /// As such, it fallback to `SetMut::__ror__` which will call `SetMut::__or__` and return a new `PySet` instead of updating the current one in-place.\
    /// Which then just doesn't work since we don't return anything, so we end up creating a new set AND then discarding it.
    fn __iand__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).intersection_update((value,))
    }
    fn __ior__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).update((value,))
    }
    fn __isub__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).difference_update((value,))
    }
    fn __ixor__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<()> {
        self.inner
            .bind(value.py())
            .symmetric_difference_update(value)
    }

    fn __sub__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .sub(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }
    fn __rsub__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        value
            .sub(self.inner.bind(value.py()))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }

    fn __xor__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .bitxor(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }
    fn __rand__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(value)
    }

    fn __ror__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__or__(value)
    }
    fn __rxor__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__xor__(value)
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).le(value)
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).lt(value)
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).ge(value)
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).gt(value)
    }

    #[staticmethod]
    fn from_ref(data: Bound<'_, PySet>) -> PyResult<Bound<'_, Self>> {
        data.into_pyochain()
    }

    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).add(value)
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get()
            .inner
            .bind(slf.py())
            .call_method0(intern!(slf.py(), "copy"))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).discard(value)?;
        Ok(())
    }
    #[pyo3(signature = (*s))]
    fn intersection_update(&self, s: Bound<'_, PyTuple>) -> PyResult<()> {
        self.inner.bind(s.py()).intersection_update(s)
    }
    fn isdisjoint<'py>(&self, s: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner.bind(s.py()).isdisjoint(s)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner.bind(other.py()).issubset(other)
    }
    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner.bind(other.py()).issuperset(other)
    }

    fn remove(&self, element: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(element.py()).remove(element)
    }

    fn symmetric_difference_update(&self, s: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(s.py()).symmetric_difference_update(s)
    }
    #[pyo3(signature = (*others))]
    fn intersection<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(others.py())
            .intersection(others)
            .and_then(Bound::into_pyochain)
    }

    #[pyo3(signature = (*others))]
    fn union<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(others.py())
            .union(others)
            .and_then(Bound::into_pyochain)
    }
    #[pyo3(signature = (*s))]
    fn update(&self, s: Bound<'_, PyTuple>) -> PyResult<()> {
        self.inner.bind(s.py()).update(s)
    }

    #[pyo3(signature = (*others))]
    fn difference<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(others.py())
            .difference(others)
            .and_then(Bound::into_pyochain)
    }
    #[pyo3(signature = (*s))]
    fn difference_update<'py>(&self, s: Bound<'py, PyTuple>) -> PyResult<()> {
        self.inner.bind(s.py()).difference_update(s)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(other.py())
            .symmetric_difference(other)
            .and_then(Bound::into_pyochain)
    }
}

#[inline]
fn set_eq<'py>(left: &Bound<'py, PyAny>, right: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
    let py = right.py();
    try_cast! {
        match right {
            Set | SetMut => left.eq(right.get().inner.bind(py)).map(Either::Left),
            PySet | PyFrozenSet => left.eq(right).map(Either::Left),
            _ => PyNotImplemented::get(py).into_bound().pipe(Ok).map(Either::Right),
        }
    }
}
