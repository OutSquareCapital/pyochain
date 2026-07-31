use crate::{
    abc::PyoCollection,
    pyo3_ext::{
        prelude::*,
        types::{PyAbstractSet, PyCmpOut, PyIterable, PyMutableSet},
    },
    traits::PyoABC,
};
use either::Either;
use pyo3::{
    BoundObject, IntoPyObjectExt,
    exceptions::PyKeyError,
    intern,
    prelude::*,
    types::{PyList, PyNotImplemented, PyType},
};
use pyochain_macros::BoundFromAny;
use tap::Pipe;
#[allow(unused)]
#[derive(BoundFromAny)]
enum IntoSetComp<'py> {
    Set(Bound<'py, PyAbstractSet>),
    Iterable(Bound<'py, PyIterable>),
    Other(Bound<'py, PyAny>),
}
fn py_from_iterable<'py>(
    slf: &Bound<'py, PyAny>,
    it: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAbstractSet>> {
    slf.call_method1(intern!(slf.py(), "_from_iterable"), (it,))
        .map(|x| unsafe { x.cast_into_unchecked::<PyAbstractSet>() })
}
#[pyclass(subclass, frozen, generic, extends=PyoCollection)]
pub struct PyoSet;
#[pymethods]
impl PyoSet {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    #[classmethod]
    fn _from_iterable<'py>(
        cls: &Bound<'py, PyType>,
        it: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((it,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
            .map_err(|e| {
                let name = cls.name().unwrap();
                let msg = format!("hint: As a `PyoSet` subclass, `{}::__init__` must accept a single `Iterable` argument.\n
                If you override it, make sure to override `PyoSet::_from_iterable` as well.", name);
                e.add_note(cls.py(), msg).unwrap();
                e
            })
    }

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
                .and_then(|x| py_from_iterable(slf.as_any(), &x))
                .map(Either::Left),
            Err(_) => PyNotImplemented::get(slf.py())
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
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
                .collect_bound::<PyList>(py)?
                .into_bound_py_any(py)
                .and_then(|x| py_from_iterable(slf.as_any(), &x))
                .map(Either::Left),
            Err(_) => PyNotImplemented::get(py)
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }

    fn __sub__<'py>(
        slf: Bound<'py, Self>,
        other: IntoSetComp<'py>,
    ) -> PyCmpOut<'py, Bound<'py, PyAbstractSet>> {
        let py = slf.py();
        let other_set = match other {
            IntoSetComp::Set(other) => other.into_any(),
            IntoSetComp::Iterable(iterable) => {
                py_from_iterable(slf.as_any(), iterable.as_any())?.into_any()
            }
            _ => {
                return PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right);
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
            .as_any()
            .pipe(|x| py_from_iterable(slf.as_any(), x))
            .map(Either::Left)
    }

    fn __xor__<'py>(
        slf: Bound<'py, Self>,
        other: IntoSetComp<'py>,
    ) -> PyCmpOut<'py, Bound<'py, PyAny>> {
        let other_set = match other {
            IntoSetComp::Set(set) => set,
            IntoSetComp::Iterable(iterable) => py_from_iterable(slf.as_any(), iterable.as_any())?,
            IntoSetComp::Other(_) => {
                return PyNotImplemented::get(slf.py())
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right);
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
            IntoSetComp::Iterable(iterable) => {
                py_from_iterable(slf.as_any(), iterable.as_any())?.into_any()
            }
            _ => {
                return PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right);
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
            .pipe(|x| py_from_iterable(slf.as_any(), &x))
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
                Ok(out).map(Either::Left)
            }
            _ => PyNotImplemented::get(slf.py())
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }
    fn __le__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if !other.is_instance_of::<PyAbstractSet>() {
            return PyNotImplemented::get(slf.py())
                .into_bound()
                .pipe(Ok)
                .map(Either::Right);
        }
        if slf.len()? > other.len()? {
            return Ok(false).map(Either::Left);
        }
        for elem in slf.try_iter()? {
            if !other.contains(elem?)? {
                return Ok(false).map(Either::Left);
            }
        }
        return Ok(true).map(Either::Left);
    }

    fn __ge__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if !other.is_instance_of::<PyAbstractSet>() {
            return PyNotImplemented::get(slf.py())
                .into_bound()
                .pipe(Ok)
                .map(Either::Right);
        }
        if slf.len()? < other.len()? {
            return Ok(false).map(Either::Left);
        }
        for elem in other.try_iter()? {
            if !slf.contains(elem?)? {
                return Ok(false).map(Either::Left);
            }
        }
        return Ok(true).map(Either::Left);
    }

    fn __lt__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if other.is_instance_of::<PyAbstractSet>() {
            Ok(slf.len()? < other.len()? && slf.le(other)?).map(Either::Left)
        } else {
            PyNotImplemented::get(slf.py())
                .into_bound()
                .pipe(Ok)
                .map(Either::Right)
        }
    }

    fn __gt__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if other.is_instance_of::<PyAbstractSet>() {
            Ok(slf.len()? > other.len()? && slf.ge(other)?).map(Either::Left)
        } else {
            PyNotImplemented::get(slf.py())
                .into_bound()
                .pipe(Ok)
                .map(Either::Right)
        }
    }

    fn isdisjoint(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        for value in other.try_iter()? {
            if slf.contains(value?)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// Mirrors `_collections_abc.Set._hash`.\
    /// Python masks with `MASK = 2 * sys.maxsize + 1` (i.e. the full 64-bit range) after every step.\
    /// This is equivalent to doing the whole computation with wrapping u64 arithmetic,\
    /// and reinterpreting the final bit pattern as a signed 64-bit hash.
    fn _hash(slf: Bound<'_, Self>) -> PyResult<isize> {
        let n = slf.len()? as u64;
        let mut h = 1927868237u64.wrapping_mul(n.wrapping_add(1));
        for x in slf.try_iter()? {
            let hx = x?.hash()? as u64;
            let mixed = hx ^ (hx << 16) ^ 89869747;
            h ^= mixed.wrapping_mul(3644798167);
        }
        h ^= (h >> 11) ^ (h >> 25);
        h = h.wrapping_mul(69069).wrapping_add(907133923);
        let h = h as isize;
        Ok(if h == -1 { 590923713 } else { h })
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

#[pyclass(subclass, frozen, generic, extends=PyoSet)]
pub struct PyoMutableSet;

#[pymethods]
impl PyoMutableSet {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn into_mutable_set(slf: Bound<'_, Self>) -> Bound<'_, PyMutableSet> {
        unsafe { slf.cast_into_unchecked::<PyMutableSet>() }
    }

    fn as_mutable_set<'a, 'py>(slf: &'a Bound<'py, Self>) -> &'a Bound<'py, PyMutableSet> {
        unsafe { slf.cast_unchecked::<PyMutableSet>() }
    }

    fn __ior__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let slf = Self::into_mutable_set(slf);
        it.try_iter()?.try_for_each(|value| slf.add(&value?))
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let slf = Self::into_mutable_set(slf);
        slf.sub(&it)?
            .try_iter()?
            .try_for_each(|value| slf.discard(&value?))
    }

    fn __isub__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let py = slf.py();
        if it.is(&slf) {
            slf.call_method0(intern!(py, "clear"))?;
        } else {
            let slf = Self::into_mutable_set(slf);
            for value in it.try_iter()? {
                slf.discard(&value?)?;
            }
        }
        Ok(())
    }

    fn __ixor__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let py = slf.py();
        if it.is(&slf) {
            slf.call_method0(intern!(py, "clear"))?;
        } else {
            let pyset = if !it.is_instance_of::<PyAbstractSet>() {
                py_from_iterable(slf.as_any(), &it)?.into_any()
            } else {
                it
            };
            let slf = Self::into_mutable_set(slf);
            for value in pyset.try_iter()? {
                let v = value?;
                if slf.contains(&v)? {
                    slf.discard(&v)?;
                } else {
                    slf.add(&v)?;
                }
            }
        }

        Ok(())
    }

    fn remove(slf: Bound<'_, Self>, value: Bound<'_, PyAny>) -> PyResult<()> {
        if !slf.contains(&value)? {
            Err(PyKeyError::new_err(format!("{}", value)))
        } else {
            Self::into_mutable_set(slf).discard(&value)?;
            Ok(())
        }
    }

    fn pop(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyAny>> {
        match PopResult::new(slf.as_any()) {
            PopResult::KeyStop => Err(PyKeyError::new_err("")),
            PopResult::Value(value) => {
                Self::into_mutable_set(slf).discard(&value)?;
                Ok(value)
            }
            PopResult::Error(e) => Err(e),
        }
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        let any = slf.as_any();
        let slf = Self::as_mutable_set(&slf);
        loop {
            match PopResult::new(any) {
                PopResult::Error(e) => {
                    return Err(e);
                }
                PopResult::KeyStop => break Ok(()),
                PopResult::Value(v) => {
                    slf.discard(&v)?;
                    continue;
                }
            }
        }
    }
}
enum PopResult<'py> {
    Value(Bound<'py, PyAny>),
    KeyStop,
    Error(PyErr),
}
impl<'py> PopResult<'py> {
    fn new(slf: &Bound<'py, PyAny>) -> Self {
        slf.try_iter()
            .map(|mut x| match x.next() {
                None => PopResult::KeyStop,
                Some(Ok(v)) => PopResult::Value(v),
                Some(Err(e)) => PopResult::Error(e),
            })
            .unwrap_or_else(|e| PopResult::Error(e))
    }
}
