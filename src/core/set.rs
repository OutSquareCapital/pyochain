use crate::{
    abc,
    display::get_repr,
    traits::{IntoPyochain, PyWrapper},
};
use either::Either;
use pyo3::{
    PyTypeInfo, intern,
    prelude::*,
    types::{DerefToPyAny, PyBool, PyFrozenSet, PyIterator, PyNotImplemented, PySet, PyTuple},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyAbstractSet, PyCmpOut},
};
use pyochain_macros::{BoundFromAny, try_cast};
use tap::Pipe;
/// Accepted types for set operations.
/// In the case of pyochain types, we extract the inner sets.
/// For python builtins, we directly work with them and call the corresponding numeric operators
/// For abstract sets, it depends on the operation.\
/// Python builtins (e.g `frozenset`) will immediately return `NotImplemented`, thus delegating to the `AbstractSet`'s corresponding operator.\
/// The issue with this is that this will cause issues and performance problems since the `AbstractSet` will himself work with another `AbstractSet` (`self` in this case).
/// Hence, we can't guarantee that the return type will be the corresponding `pyochain` type, thus leaving us with two choices:
/// 1. Runtime check, no `cast_into_unchecked`, thus performance impact, and we still need to handle the return type.
/// 2. (What we do) -> We call the `AbstractSet`'s corresponding operator, and then we try to iterate over the result and collect it into the corresponding `pyochain` type.\
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
trait SetCmpMethods<
    'py,
    T: PyTypeInfo + DerefToPyAny + TryFromBoundIterator<'py, Bound<'py, PyAny>>,
>: Sized + PyWrapper + PyTypeInfo
{
    #[inline(always)]
    fn handle_pyabstract_set(pyset: Bound<'py, PyAny>) -> PyResult<Bound<'py, T>> {
        let py = pyset.py();
        match pyset.cast_exact::<T>() {
            Err(_) => pyset.try_iter()?.try_collect_bound::<T>(py),
            Ok(target) => Ok(target.into()),
        }
    }
    #[inline]
    fn cmp_eq(&self, right: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        let py = right.py();
        let inner = self.inner().bind(py);
        try_cast! {
            match right {
                CaseExact::Set(set) | CaseExact::SetMut(set) => inner.eq(set.get().inner_bind(py)).map(Either::Left),
                Case::PySet(pyset) | Case::PyFrozenSet(pyset) => inner.eq(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    #[inline]
    fn cmp_and(&self, other: SetCmp<'py>) -> PyResult<Bound<'py, T>> {
        let inner = self.inner();
        match &other {
            SetCmp::Set(x) => inner
                .bind(x.py())
                .bitand(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::SetMut(x) => inner
                .bind(x.py())
                .bitand(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PySet(x) => inner
                .bind(x.py())
                .bitand(x)
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PyFrozen(x) => inner
                .bind(x.py())
                .bitand(x)
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PyAbstract(x) => Self::handle_pyabstract_set(x.bitand(inner)?),
        }
    }

    #[inline]
    fn cmp_or(&self, other: SetCmp<'py>) -> PyResult<Bound<'py, T>> {
        let inner = self.inner();
        match &other {
            SetCmp::Set(x) => inner
                .bind(x.py())
                .bitor(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::SetMut(x) => inner
                .bind(x.py())
                .bitor(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PySet(x) => inner
                .bind(x.py())
                .bitor(x)
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PyFrozen(x) => inner
                .bind(x.py())
                .bitor(x)
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PyAbstract(x) => Self::handle_pyabstract_set(x.bitor(inner)?),
        }
    }

    #[inline]
    fn cmp_xor(&self, other: SetCmp<'py>) -> PyResult<Bound<'py, T>> {
        let inner = self.inner();
        match &other {
            SetCmp::Set(x) => inner
                .bind(x.py())
                .bitxor(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::SetMut(x) => inner
                .bind(x.py())
                .bitxor(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PySet(x) => inner
                .bind(x.py())
                .bitxor(x)
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PyFrozen(x) => inner
                .bind(x.py())
                .bitxor(x)
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PyAbstract(x) => Self::handle_pyabstract_set(x.bitxor(inner)?),
        }
    }

    #[inline]
    fn cmp_sub(&self, other: SetCmp<'py>) -> PyResult<Bound<'py, T>> {
        let inner = self.inner();
        match &other {
            SetCmp::Set(x) => inner
                .bind(x.py())
                .sub(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::SetMut(x) => inner
                .bind(x.py())
                .sub(x.get().inner_bind(x.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PySet(x) => inner
                .bind(x.py())
                .sub(x)
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PyFrozen(x) => inner
                .bind(x.py())
                .sub(x)
                .map(|x| unsafe { x.cast_into_unchecked::<T>() }),
            SetCmp::PyAbstract(x) => Self::handle_pyabstract_set(inner.bind(x.py()).sub(x)?),
        }
    }
}
impl<'py> SetCmpMethods<'py, PyFrozenSet> for Set {}
impl<'py> SetCmpMethods<'py, PySet> for SetMut {}
#[pyclass(module = "pyochain.core",frozen, generic, extends=abc::PyoSet)]
pub struct Set(pub Py<PyFrozenSet>);
#[pymethods]
impl Set {
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
        self.inner_bind(item.py()).contains(item)
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner_bind(slf.py()).iter_py()
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        slf.get().inner_bind(slf.py()).len()
    }

    fn __and__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.cmp_and(value).and_then(Bound::into_pyochain)
    }
    fn __or__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.cmp_or(value).and_then(Bound::into_pyochain)
    }

    fn __sub__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.cmp_sub(value).and_then(Bound::into_pyochain)
    }

    fn __xor__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.cmp_xor(value).and_then(Bound::into_pyochain)
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
        self.cmp_eq(value)
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
#[pyclass(module = "pyochain.core",frozen, generic, extends=abc::PyoMutableSet)]
pub struct SetMut(pub Py<PySet>);
#[pymethods]
impl SetMut {
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner_bind(slf.py()).iter_py()
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
        self.cmp_eq(other)
    }

    fn __and__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.cmp_and(value).and_then(Bound::into_pyochain)
    }

    fn __or__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.cmp_or(value).and_then(Bound::into_pyochain)
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
        self.cmp_sub(value).and_then(Bound::into_pyochain)
    }

    fn __xor__<'py>(&self, value: SetCmp<'py>) -> PyResult<Bound<'py, Self>> {
        self.cmp_xor(value).and_then(Bound::into_pyochain)
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

    fn remove(&self, element: &Bound<'_, PyAny>) -> PyResult<()> {
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
