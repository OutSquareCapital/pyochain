use crate::{
    abc,
    display::get_repr,
    pyo3_ext::{
        prelude::*,
        types::{PyAbstractSet, PyCmpOut},
    },
    traits::{IntoPyochain, PyWrapper, PyoABC},
};
use either::Either;
use pyo3::{
    BoundObject, PyTypeInfo, intern,
    prelude::*,
    pyclass_init::PyClassInitializer,
    types::{PyBool, PyFrozenSet, PyIterator, PyNotImplemented, PySet, PyTuple},
};
use pyochain_macros::{BoundFromAny, try_cast};
use tap::Pipe;
#[derive(BoundFromAny)]
enum SetCmp<'py> {
    #[cast_exact]
    PyFrozen(Bound<'py, PyFrozenSet>),
    #[cast_exact]
    PySet(Bound<'py, PySet>),
    #[cast_exact]
    Set(Bound<'py, Set>),
    #[cast_exact]
    SetMut(Bound<'py, SetMut>),
    PyAbstract(Bound<'py, PyAbstractSet>),
}
#[pyclass(frozen, generic, extends=abc::PyoSet)]
pub struct Set(pub Py<PyFrozenSet>);
#[pymethods]
impl Set {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        data.pipe(|x| PyFrozenSet::type_object(py).call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
            .map(Bound::unbind)
            .map(|inner| abc::PyoSet::build_init().add_subclass(Self(inner)))
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;
        slf.get()
            .inner_bind(py)
            .pipe(|x| PyTuple::new(py, x))
            .and_then(get_repr)
            .map(|repr| format!("{}({})", name, repr))
    }

    fn __contains__(&self, item: Bound<'_, PyAny>) -> PyResult<bool> {
        return self.inner_bind(item.py()).contains(item);
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        return slf.get().inner_bind(slf.py()).try_iter().unwrap();
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        return slf.get().inner_bind(slf.py()).len();
    }

    fn __and__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        let inner = &self.inner();
        match &value {
            SetCmp::Set(x) => inner.bind(x.py()).bitand(x.get().inner_bind(x.py())),
            SetCmp::SetMut(x) => inner.bind(x.py()).bitand(x.get().inner_bind(x.py())),
            SetCmp::PySet(x) => inner.bind(x.py()).bitand(x),
            SetCmp::PyFrozen(x) => inner.bind(x.py()).bitand(x),

            SetCmp::PyAbstract(x) => inner.bind(x.py()).bitand(x),
        }
        .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
        .and_then(Bound::into_pyochain)
    }
    fn __or__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        let inner = &self.inner();
        match &value {
            SetCmp::Set(x) => inner.bind(x.py()).bitor(x.get().inner_bind(x.py())),
            SetCmp::SetMut(x) => inner.bind(x.py()).bitor(x.get().inner_bind(x.py())),
            SetCmp::PySet(x) => inner.bind(x.py()).bitor(x),
            SetCmp::PyFrozen(x) => inner.bind(x.py()).bitor(x),
            SetCmp::PyAbstract(x) => inner.bind(x.py()).bitor(x),
        }
        .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
        .and_then(Bound::into_pyochain)
    }

    fn __sub__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        let inner = &self.inner();
        let sub_into_set = |x: Bound<'py, PyAny>| {
            inner
                .bind(x.py())
                .sub(x)
                .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
        };
        match value {
            SetCmp::Set(x) => sub_into_set(x.get().into_inner_bound(x.py()).into_any()),
            SetCmp::SetMut(x) => sub_into_set(x.get().into_inner_bound(x.py()).into_any()),
            SetCmp::PySet(x) => sub_into_set(x.into_any()),
            SetCmp::PyFrozen(x) => sub_into_set(x.into_any()),
            SetCmp::PyAbstract(x) => {
                let py = x.py();
                inner
                    .bind(py)
                    .sub(x)?
                    .try_iter()?
                    .collect_bound::<PyFrozenSet>(py)
            }
        }
        .and_then(Bound::into_pyochain)
    }

    fn __xor__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        let inner = &self.inner();
        match &value {
            SetCmp::Set(x) => inner.bind(x.py()).bitxor(x.get().inner_bind(x.py())),
            SetCmp::SetMut(x) => inner.bind(x.py()).bitxor(x.get().inner_bind(x.py())),
            SetCmp::PySet(x) => inner.bind(x.py()).bitxor(x),
            SetCmp::PyFrozen(x) => inner.bind(x.py()).bitxor(x),
            SetCmp::PyAbstract(x) => inner.bind(x.py()).bitxor(x),
        }
        .map(|x| unsafe { x.cast_into_unchecked::<PyFrozenSet>() })
        .and_then(Bound::into_pyochain)
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).le(value)
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).lt(value)
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).ge(value)
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).gt(value)
    }

    fn __eq__<'py>(&self, value: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        self.inner_bind(value.py())
            .as_any()
            .pipe_ref(|x| set_eq(x, value))
    }

    fn __hash__(slf: Bound<'_, Self>) -> PyResult<isize> {
        slf.get().inner_bind(slf.py()).hash()
    }

    fn __rand__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.__and__(value)
    }
    fn __ror__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.__or__(value)
    }
    fn __rsub__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.__sub__(value)
    }
    fn __rxor__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.__xor__(value)
    }

    fn isdisjoint<'py>(&self, s: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner_bind(s.py()).isdisjoint(s)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner_bind(other.py()).issubset(other)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner_bind(other.py()).issuperset(other)
    }
    #[pyo3(signature = (*others))]
    fn intersection<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(others.py())
            .intersection(others)
            .and_then(Bound::into_pyochain)
    }

    #[pyo3(signature = (*others))]
    fn union<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(others.py())
            .union(others)
            .and_then(Bound::into_pyochain)
    }

    #[pyo3(signature = (*others))]
    fn difference<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(others.py())
            .difference(others)
            .and_then(Bound::into_pyochain)
    }

    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(other.py())
            .symmetric_difference(other)
            .and_then(Bound::into_pyochain)
    }
}
#[pyclass(frozen, generic, extends=abc::PyoMutableSet)]
pub struct SetMut(pub Py<PySet>);
#[pymethods]
impl SetMut {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        data.pipe(|x| PySet::type_object(py).call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .map(Bound::unbind)
            .map(|inner| abc::PyoMutableSet::build_init().add_subclass(SetMut(inner)))
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner_bind(slf.py()).try_iter().unwrap()
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        slf.get().inner_bind(slf.py()).len()
    }

    fn __contains__(&self, item: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(item.py()).contains(item)
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;
        slf.get()
            .inner_bind(py)
            .pipe(|x| PyTuple::new(py, x))
            .and_then(get_repr)
            .map(|repr| format!("{}({})", name, repr))
    }

    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        self.inner_bind(other.py())
            .as_any()
            .pipe_ref(|x| set_eq(x, other))
    }

    fn __and__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        let inner = &self.inner();

        match &value {
            SetCmp::Set(x) => inner.bind(x.py()).bitand(x.get().inner_bind(x.py())),
            SetCmp::SetMut(x) => inner.bind(x.py()).bitand(x.get().inner_bind(x.py())),
            SetCmp::PySet(x) => inner.bind(x.py()).bitand(x),
            SetCmp::PyFrozen(x) => inner.bind(x.py()).bitand(x),
            SetCmp::PyAbstract(x) => inner.bind(x.py()).bitand(x),
        }
        .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
        .and_then(Bound::into_pyochain)
    }

    fn __or__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        let inner = &self.inner();
        match &value {
            SetCmp::Set(x) => inner.bind(x.py()).bitor(x.get().inner_bind(x.py())),
            SetCmp::SetMut(x) => inner.bind(x.py()).bitor(x.get().inner_bind(x.py())),
            SetCmp::PySet(x) => inner.bind(x.py()).bitor(x),
            SetCmp::PyFrozen(x) => inner.bind(x.py()).bitor(x),
            SetCmp::PyAbstract(x) => inner.bind(x.py()).bitor(x),
        }
        .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
        .and_then(Bound::into_pyochain)
    }
    /// NOTE: We need to use `call_method1` for in-place operators here because Pyo3 doesn't allow returning something else than `PyResult<()>`.\
    /// And if we use `PySet::__ior__`, it will return `NotImplemented` on object who are NOT subclasses of `set` or `frozenset`.\
    /// As such, it fallback to `SetMut::__ror__` which will call `SetMut::__or__` and return a new `PySet` instead of updating the current one in-place.\
    /// Which then just doesn't work since we don't return anything, so we end up creating a new set AND then discarding it.
    fn __iand__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).intersection_update((value,))
    }
    fn __ior__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).update((value,))
    }
    fn __isub__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).difference_update((value,))
    }
    fn __ixor__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py())
            .symmetric_difference_update(value)
    }

    fn __sub__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        let inner = &self.inner();
        match &value {
            SetCmp::Set(x) => inner
                .bind(x.py())
                .sub(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<PySet>() }),
            SetCmp::SetMut(x) => inner
                .bind(x.py())
                .sub(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<PySet>() }),
            SetCmp::PySet(x) => inner
                .bind(x.py())
                .sub(x)
                .map(|x| unsafe { x.cast_into_unchecked::<PySet>() }),
            SetCmp::PyFrozen(x) => inner
                .bind(x.py())
                .sub(x)
                .map(|x| unsafe { x.cast_into_unchecked::<PySet>() }),
            SetCmp::PyAbstract(x) => {
                let py = x.py();
                inner
                    .bind(py)
                    .sub(x)?
                    .try_iter()?
                    .collect_bound::<PySet>(py)
            }
        }
        .and_then(Bound::into_pyochain)
    }

    fn __xor__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        let inner = &self.inner();
        match &value {
            SetCmp::Set(x) => inner.bind(x.py()).bitxor(x.get().inner_bind(x.py())),
            SetCmp::SetMut(x) => inner.bind(x.py()).bitxor(x.get().inner_bind(x.py())),
            SetCmp::PySet(x) => inner.bind(x.py()).bitxor(x),
            SetCmp::PyFrozen(x) => inner.bind(x.py()).bitxor(x),
            SetCmp::PyAbstract(x) => inner.bind(x.py()).bitxor(x),
        }
        .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
        .and_then(Bound::into_pyochain)
    }

    fn __rand__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.__and__(value)
    }
    fn __ror__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.__or__(value)
    }
    fn __rsub__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.__sub__(value)
    }
    fn __rxor__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.__xor__(value)
    }
    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).le(value)
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).lt(value)
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).ge(value)
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(value.py()).gt(value)
    }

    #[staticmethod]
    fn from_ref(data: Bound<'_, PySet>) -> PyResult<Bound<'_, Self>> {
        data.into_pyochain()
    }

    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).add(value)
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get()
            .inner_bind(slf.py())
            .call_method0(intern!(slf.py(), "copy"))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).discard(value)?;
        Ok(())
    }
    #[pyo3(signature = (*s))]
    fn intersection_update(&self, s: Bound<'_, PyTuple>) -> PyResult<()> {
        self.inner_bind(s.py()).intersection_update(s)
    }
    fn isdisjoint<'py>(&self, s: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner_bind(s.py()).isdisjoint(s)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner_bind(other.py()).issubset(other)
    }
    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.inner_bind(other.py()).issuperset(other)
    }

    fn remove(&self, element: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(element.py()).remove(element)
    }

    fn symmetric_difference_update(&self, s: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(s.py()).symmetric_difference_update(s)
    }
    #[pyo3(signature = (*others))]
    fn intersection<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(others.py())
            .intersection(others)
            .and_then(Bound::into_pyochain)
    }

    #[pyo3(signature = (*others))]
    fn union<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(others.py())
            .union(others)
            .and_then(Bound::into_pyochain)
    }
    #[pyo3(signature = (*s))]
    fn update(&self, s: Bound<'_, PyTuple>) -> PyResult<()> {
        self.inner_bind(s.py()).update(s)
    }

    #[pyo3(signature = (*others))]
    fn difference<'py>(&self, others: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(others.py())
            .difference(others)
            .and_then(Bound::into_pyochain)
    }
    #[pyo3(signature = (*s))]
    fn difference_update<'py>(&self, s: Bound<'py, PyTuple>) -> PyResult<()> {
        self.inner_bind(s.py()).difference_update(s)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(other.py())
            .symmetric_difference(other)
            .and_then(Bound::into_pyochain)
    }
}

#[inline]
fn set_eq<'py>(left: &Bound<'py, PyAny>, right: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
    let py = right.py();
    try_cast! {
        match right {
            Set | SetMut => left.eq(right.get().inner_bind(py)).map(Either::Left),
            PySet | PyFrozenSet => left.eq(right).map(Either::Left),
            _ => PyNotImplemented::get(py).into_bound().pipe(Ok).map(Either::Right),
        }
    }
}
