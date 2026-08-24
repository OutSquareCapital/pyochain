use crate::{abc::PyoCollection, collections::sorted, hasher};
use either::Either;
use pyo3::{
    PyClass, PyTypeInfo, intern,
    prelude::*,
    types::{DerefToPyAny, PyList, PyNotImplemented, PyType},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyAbstractSet, PyCmpOut, PyIterable},
};
use pyochain_macros::{BoundFromAny, py_abc};
use tap::Pipe;
#[py_abc(
    PyoSet,
    sorted::SortedKeysView,
    sorted::SortedByKeyKeysView,
    sorted::SortedItemsView,
    sorted::SortedByKeyItemsView
)]
pub trait PyoSetMethods: PyClass + PyTypeInfo + DerefToPyAny {
    fn __and__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyCmpOut<'py, Bound<'py, PyAbstractSet>> {
        let py = other.py();
        match other.try_iter() {
            Ok(mut iterator) => iterator
                .try_fold(PyList::empty(py), |acc, value| {
                    let v = value?;
                    if slf.contains(&v)? {
                        acc.append(v)?;
                    }
                    Ok(acc)
                })
                .and_then(|x| py_from_iterable(&slf, &x))
                .map(Either::Left),
            Err(_) => PyNotImplemented::from_cmp(py),
        }
    }
    fn __or__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyCmpOut<'py, Bound<'py, PyAbstractSet>> {
        let py = slf.py();
        match other.try_iter() {
            Ok(iterator) => slf
                .try_iter()?
                .chain(iterator)
                .try_collect_bound::<PyList>(py)?
                .pipe(|x| py_from_iterable(&slf, &x))
                .map(Either::Left),
            Err(_) => PyNotImplemented::from_cmp(py),
        }
    }

    fn __sub__<'py>(
        slf: Bound<'py, Self>,
        other: IntoSetComp<'py>,
    ) -> PyCmpOut<'py, Bound<'py, PyAbstractSet>> {
        let py = slf.py();
        let other_set = match other {
            IntoSetComp::Set(other) => other.into_any(),
            IntoSetComp::Iterable(iterable) => py_from_iterable(&slf, &iterable)?.into_any(),
            _ => {
                return PyNotImplemented::from_cmp(py);
            }
        };
        slf.try_iter()?
            .try_fold(PyList::empty(py), |init, x| {
                let item = x?;
                if !other_set.contains(&item)? {
                    init.append(item)?;
                }
                Ok::<_, PyErr>(init)
            })?
            .pipe(|x| py_from_iterable(&slf, &x))
            .map(Either::Left)
    }

    fn __xor__<'py>(
        slf: Bound<'py, Self>,
        other: IntoSetComp<'py>,
    ) -> PyCmpOut<'py, Bound<'py, PyAny>> {
        let other_set = match other {
            IntoSetComp::Set(set) => set,
            IntoSetComp::Iterable(iterable) => py_from_iterable(&slf, &iterable)?,
            IntoSetComp::Other(_) => {
                return PyNotImplemented::from_cmp(slf.py());
            }
        };
        slf.sub(&other_set)?
            .bitor(&other_set.sub(slf)?)
            .map(Either::Left)
    }

    fn __rand__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        slf.bitand(other)
    }

    fn __rsub__<'py>(
        slf: Bound<'py, Self>,
        other: IntoSetComp<'py>,
    ) -> PyCmpOut<'py, Bound<'py, PyAbstractSet>> {
        let py = slf.py();
        let other_set = match other {
            IntoSetComp::Set(other) => other.into_any(),
            IntoSetComp::Iterable(iterable) => py_from_iterable(&slf, &iterable)?.into_any(),
            _ => {
                return PyNotImplemented::from_cmp(py);
            }
        };
        other_set
            .try_iter()?
            .try_fold(PyList::empty(py), |init, x| {
                let item = x?;
                if !slf.contains(&item)? {
                    init.append(item)?;
                }
                Ok::<_, PyErr>(init)
            })?
            .pipe(|x| py_from_iterable(&slf, &x))
            .map(Either::Left)
    }

    fn __ror__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        slf.bitor(other)
    }
    fn __rxor__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        slf.bitxor(other)
    }
    fn __eq__<'py>(slf: Bound<'py, Self>, other: IntoSetComp<'py>) -> PyCmpOut<bool, 'py> {
        match other {
            IntoSetComp::Set(other_set) => {
                let out = slf.len()? == other_set.len()? && slf.le(&other_set)?;
                Ok(Either::Left(out))
            }
            _ => PyNotImplemented::from_cmp(slf.py()),
        }
    }
    fn __le__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if !other.is_instance_of::<PyAbstractSet>() {
            return PyNotImplemented::from_cmp(slf.py());
        }
        if slf.len()? > other.len()? {
            return Ok(Either::Left(false));
        }
        for elem in slf.try_iter()? {
            if !other.contains(elem?)? {
                return Ok(Either::Left(false));
            }
        }
        Ok(Either::Left(true))
    }

    fn __ge__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if !other.is_instance_of::<PyAbstractSet>() {
            return PyNotImplemented::from_cmp(slf.py());
        }
        if slf.len()? < other.len()? {
            return Ok(Either::Left(false));
        }
        for elem in other.try_iter()? {
            if !slf.contains(elem?)? {
                return Ok(Either::Left(false));
            }
        }
        Ok(Either::Left(true))
    }

    fn __lt__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if other.is_instance_of::<PyAbstractSet>() {
            Ok(Either::Left(slf.len()? < other.len()? && slf.le(other)?))
        } else {
            PyNotImplemented::from_cmp(slf.py())
        }
    }

    fn __gt__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if other.is_instance_of::<PyAbstractSet>() {
            Ok(Either::Left(slf.len()? > other.len()? && slf.ge(other)?))
        } else {
            PyNotImplemented::from_cmp(slf.py())
        }
    }
    fn _hash(slf: Bound<'_, Self>) -> PyResult<isize> {
        slf.pipe_ref(hasher::set_hash)
    }
    fn isdisjoint(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        for value in other.try_iter()? {
            if slf.contains(value?)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn is_subset(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.le(other)
    }
    fn is_subset_strict(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.lt(other)
    }
    fn eq(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.eq(other)
    }
    fn is_superset(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.ge(other)
    }
    fn is_superset_strict(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.gt(other)
    }
    fn is_disjoint(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        Self::isdisjoint(slf, other)
    }
    fn intersection<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.bitand(other)
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        slf.bitor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.sub(other)
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn symmetric_difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.bitxor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
}
impl PyoSetMethods for PyoSet {}
impl PyoSetMethods for sorted::SortedKeysView {}
impl PyoSetMethods for sorted::SortedByKeyKeysView {}
impl PyoSetMethods for sorted::SortedItemsView {}
impl PyoSetMethods for sorted::SortedByKeyItemsView {}

#[allow(unused)]
#[derive(BoundFromAny)]
pub enum IntoSetComp<'py> {
    Set(Bound<'py, PyAbstractSet>),
    Iterable(Bound<'py, PyIterable>),
    Other(Bound<'py, PyAny>),
}
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoCollection)]
pub struct PyoSet;
#[pymethods]
impl PyoSet {
    #[classmethod]
    fn _from_iterable<'py>(
        cls: &Bound<'py, PyType>,
        it: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((it,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
            .inspect_err(|e| {
                let name = cls.name().unwrap();
                let msg = format!("hint: As a `PyoSet` subclass, `{}::__init__` must accept a single `Iterable` argument.\n
                If you override it, make sure to override `PyoSet::_from_iterable` as well.", name);
                e.add_note(cls.py(), msg).unwrap();
            })
    }
}

pub(super) fn py_from_iterable<'py, T: DerefToPyAny, U: DerefToPyAny + PyTypeInfo>(
    slf: &Bound<'py, T>,
    it: &Bound<'py, U>,
) -> PyResult<Bound<'py, PyAbstractSet>> {
    slf.call_method1(intern!(slf.py(), "_from_iterable"), (it,))
        .map(|x| unsafe { x.cast_into_unchecked::<PyAbstractSet>() })
}
