use pyo3::{IntoPyObjectExt, exceptions::{PyKeyError, PyNotImplementedError}, intern, prelude::*, types::{PyList, PyType}};
use crate::{abc::PyoABC, pyo3_ext::args::Args};
use crate::{abc::PyoCollection, pyo3_ext::{args::Kwargs, types::PyAbstractSet}};
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
                let msg = format!("hint: As a `PyoSet` subclass, `{}::__init__` must accept a single `Iterable` argument. If you override it, make sure to override `PyoSet::_from_iterable` as well.", name);
                e.add_note(cls.py(), msg,).unwrap(); 
                e})
    
}
    
    #[inline]
    #[classmethod]
    fn _py_from_iterable<'py>(
        cls: &Bound<'py, PyType>,
        it: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call_method1(intern!(cls.py(), "_from_iterable"), (it,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    fn __and__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if !other.is_instance_of::<PyAbstractSet>() {
            return Err(PyNotImplementedError::new_err(""));
        }
        slf.try_iter()?
            .try_fold(PyList::empty(py), |init, x| {
                let item = x?;
                if other.contains(&item)? {
                    init.append(item)?;
                }
                Ok::<_, PyErr>(init)
            })?
            .into_bound_py_any(py)
            .and_then(|x| Self::_py_from_iterable(&slf.get_type(), &x))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn __or__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if !other.is_instance_of::<PyAbstractSet>() {
            return Err(PyNotImplementedError::new_err(""));
        }
        slf.try_iter()?
            .chain(other.try_iter()?)
            .try_fold(PyList::empty(py), |init, x| {
                let item = x?;
                init.append(item)?;
                Ok::<_, PyErr>(init)
            })?
            .into_bound_py_any(py)
            .and_then(|x| Self::_py_from_iterable(&slf.get_type(), &x))
    }

    fn __sub__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if !other.is_instance_of::<PyAbstractSet>() {
            return Err(PyNotImplementedError::new_err(""));
        }
        other
            .try_iter()
            .map_err(|_| PyNotImplementedError::new_err(""))
            .and_then(|iterator| {
                let cls = slf.get_type();
                let other_set = Self::_py_from_iterable(&cls, iterator.as_any())?;
                slf.try_iter()?
                    .try_fold(PyList::empty(py), |init, x| {
                        let item = x?;
                        if !other_set.contains(&item)? {
                            init.append(item)?;
                        }
                        Ok::<_, PyErr>(init)
                    })?
                    .into_bound_py_any(py)
                    .and_then(|x| Self::_py_from_iterable(&cls, &x))
            })
    }

    fn __rsub__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let cls = slf.get_type();
        other
            .try_iter()
            .map_err(|_| PyNotImplementedError::new_err(""))
            .and_then(|x| {
                if !other.is_instance_of::<PyAbstractSet>() {
                    Self::_py_from_iterable(&cls, x.as_any())?.try_iter()
                } else {
                    Ok(x)
                }
            })?
            .try_fold(PyList::empty(py), |init, x| {
                let item = x?;
                if !slf.contains(&item)? {
                    init.append(item)?;
                }
                Ok::<_, PyErr>(init)
            })?
            .into_bound_py_any(py)
            .and_then(|x| Self::_py_from_iterable(&cls, &x))
    }

    fn __xor__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        if other.is_instance_of::<PyAbstractSet>() {
            other
                .try_iter()
                .map_err(|_| PyNotImplementedError::new_err(""))
                .and_then(|iterator| Self::_py_from_iterable(&slf.get_type(), iterator.as_any()))
                .and_then(|x| (slf.sub(&x))?.bitor((x).sub(slf)?))
                .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
        } else {
            Err(PyNotImplementedError::new_err(""))
        }
    }

    fn __rand__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::__and__(slf, other)
    }
    fn __ror__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        Self::__or__(slf, other)
    }
    fn __rxor__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::__xor__(slf, other)
    }
    fn __eq__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_instance_of::<PyAbstractSet>() {
            Ok(slf.len()? == other.len()? && slf.le(other)?)
        } else {
            Err(PyNotImplementedError::new_err(""))
        }
    }
    fn __le__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        if !other.is_instance_of::<PyAbstractSet>() {
            return Err(PyNotImplementedError::new_err(""));
        }
        if slf.len()? > other.len()? {
            return Ok(false);
        }
        for elem in slf.try_iter()? {
            if !other.contains(elem?)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    fn __ge__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        if !other.is_instance_of::<PyAbstractSet>() {
            return Err(PyNotImplementedError::new_err(""));
        }
        if slf.len()? < other.len()? {
            return Ok(false);
        }
        for elem in other.try_iter()? {
            if !slf.contains(elem?)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    fn __lt__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_instance_of::<PyAbstractSet>() {
            Ok(slf.len()? < other.len()? && slf.le(other)?)
        } else {
            Err(PyNotImplementedError::new_err(""))
        }
    }

    fn __gt__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        if other.is_instance_of::<PyAbstractSet>() {
            Ok(slf.len()? > other.len()? && slf.ge(other)?)
        } else {
            Err(PyNotImplementedError::new_err(""))
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
    fn _hash(slf: Bound<'_, Self>) -> PyResult<isize> {
        let max = isize::MAX / 2;
        let mask = 2 * max + 1;
        let n = slf.len()? as isize;
        let mut h = 1927868237 * (n + 1);
        h &= mask;
        for x in slf.try_iter()? {
            let hx = x?.hash()?;
            h ^= (hx ^ (hx << 16) ^ 89869747) * 3644798167;
            h &= mask;
        }
        h ^= (h >> 11) ^ (h >> 25);
        h = h * 69069 + 907133923;
        h &= mask;
        if h > max {
            h -= mask + 1;
        }
        if h == -1 {
            h = 590923713;
        }
        Ok(h)
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
    fn _py_add(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method1(intern!(slf.py(), "add"), (value,))?;
        Ok(())
    }
    fn _py_discard(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method1(intern!(slf.py(), "discard"), (value,))?;
        Ok(())
    }

    fn __ior__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        it.try_iter()?
            .try_for_each(|value| Self::_py_add(&slf, &value?))
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        slf.sub(&it)?
            .try_iter()?
            .try_for_each(|value| Self::_py_discard(&slf, &value?))
    }

    fn __isub__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let py = slf.py();
        if it.is(&slf) {
            slf.call_method0(intern!(py, "clear"))?;
        } else {
            for value in it.try_iter()? {
                Self::_py_discard(&slf, &value?)?;
            }
        }
        Ok(())
    }

    fn __ixor__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let cls = slf.get_type();
        if it.is(&slf) {
            slf.call_method0(intern!(py, "clear"))?;
        } else {
            let pyset = if !it.is_instance_of::<PyAbstractSet>() {
                PyoSet::_py_from_iterable(&cls, &it)?.into_any()
            } else {
                it
            };
            for value in pyset.try_iter()? {
                let v = value?;
                if slf.contains(&v)? {
                    Self::_py_discard(&slf, &v)?;
                } else {
                    Self::_py_add(&slf, &v)?;
                }
            }
        }

        Ok(())
    }


    fn remove(slf: Bound<'_, Self>, value: Bound<'_, PyAny>) -> PyResult<()> {
        if !slf.contains(&value)? {
            Err(PyKeyError::new_err(format!("{}", value)))
        } else {
            Self::_py_discard(&slf, &value)?;
            Ok(())
        }
    }

    fn pop(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyAny>> {
        match slf.try_iter()?.next() {
            None => Err(PyKeyError::new_err("")),
            Some(value) => value.and_then(|x| {
                Self::_py_discard(&slf, &x)?;
                Ok(x)
            }),
        }
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        slf.try_iter()?
            .try_for_each(|x| Self::_py_discard(&slf, &x?))
    }
}