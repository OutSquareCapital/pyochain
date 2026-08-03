use crate::{
    abc::{self, traits::ImplPyoReversible},
    iterators,
    pyo3_ext::{
        prelude::*,
        pylibs,
        types::{PyCmpOut, PySupportsItems, pyitertools},
    },
    traits::{PyWrapper, PyoABC},
};
use either::Either;
use pyo3::{
    BoundObject, PyTypeInfo,
    exceptions::{PyKeyError, PyTypeError},
    intern,
    prelude::*,
    types::{PyDict, PyInt, PyIterator, PyList, PyMapping, PyNotImplemented, PyType},
};
use pyochain_macros::{BoundFromAny, try_cast};
use tap::prelude::*;
#[derive(BoundFromAny)]
enum IntoUpdate<'py> {
    Dict(Bound<'py, PyDict>),
    Mapping(Bound<'py, PyMapping>),
    Iterable(Bound<'py, PyAny>),
}
#[pyclass(module = "pyochain._collections",frozen, generic, mapping, extends = abc::PyoMutableMapping)]
pub struct PyoCounter(pub Py<PyDict>);
#[pymethods]
impl PyoCounter {
    #[new]
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn new(
        py: Python<'_>,
        iterable: Option<IntoUpdate<'_>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let inner = PyDict::new(py);

        update_counter(&inner, iterable, kwargs)?;
        Ok(abc::PyoMutableMapping::build_init().add_subclass(Self(inner.unbind())))
    }
    #[staticmethod]
    fn from_ref(data: Bound<'_, PyDict>) -> PyResult<Bound<'_, Self>> {
        let py = data.py();
        let initializer = abc::PyoMutableMapping::build_init().add_subclass(Self(data.unbind()));
        Bound::new(py, initializer)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner_bind(py).try_iter().unwrap()
    }

    fn __len__<'py>(&self, py: Python<'py>) -> usize {
        self.inner_bind(py).len()
    }

    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        match self.inner_bind(py).as_any().get_item(key) {
            Ok(value) => Ok(value),
            Err(err) => {
                if err.matches(py, PyKeyError::type_object(py))? {
                    Ok(PyInt::new(py, 0).into_any())
                } else {
                    Err(err)
                }
            }
        }
    }

    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyInt>) -> PyResult<()> {
        self.inner_bind(key.py()).set_item(key, value)
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(key.py()).contains(key)
    }
    #[allow(unused)]
    fn __missing__(&self, key: &Bound<'_, PyAny>) -> isize {
        0
    }
    #[pyo3(signature = (key, default=None, /))]
    fn get<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner_bind(key.py())
            .get_item(key)?
            .or(default)
            .pipe(Ok)
    }

    #[pyo3(signature = (key, default, /))]
    fn setdefault<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        self.inner_bind(py)
            .call_method1(intern!(py, "setdefault"), (key, default))
    }

    fn total<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner_bind(py)
            .values_view()
            .as_any()
            .try_iter()
            .unwrap()
            .pipe(|vals| pylibs::builtins::sum(&vals, &0))
    }
    #[pyo3(signature = (n=None))]
    fn most_common<'py>(
        &self,
        py: Python<'py>,
        n: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let items = self.inner_bind(py).items_view().try_iter().unwrap();
        let getter = pylibs::operator::itemgetter(py, 1)?;
        match n {
            None => pylibs::builtins::sorted_by(&items, true, &getter),
            Some(n) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "key"), getter)?;
                py.import(intern!(py, "heapq"))?
                    .getattr(intern!(py, "nlargest"))?
                    .call((n, items), Some(&kwargs))
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            }
        }
    }

    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>> {
        pylibs::itertools::map_star(
            pyitertools::PyRepeat::type_object(py).into_any(),
            self.inner_bind(py).items_view().try_iter().unwrap(),
        )?
        .pipe_ref(pylibs::itertools::chain::from_iterable)
        .and_then(iterators::Iter::new)
    }
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn update(
        &self,
        py: Python<'_>,
        iterable: Option<IntoUpdate<'_>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<()> {
        let inner = self.inner_bind(py);
        update_counter(&inner, iterable, kwargs)
    }
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn subtract(
        &self,
        py: Python<'_>,
        iterable: Option<IntoUpdate<'_>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<()> {
        let inner = self.inner_bind(py);
        subtract_counter(inner, iterable, kwargs)
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get_type()
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn __reduce__(slf: Bound<'_, Self>) -> (Bound<'_, PyType>, (Py<PyDict>,)) {
        let py = slf.py();
        return (Self::type_object(py), (slf.get().inner().clone_ref(py),));
    }

    fn __delitem__(&self, elem: Bound<'_, PyAny>) -> PyResult<()> {
        let inner = self.inner_bind(elem.py());
        if inner.contains(&elem)? {
            inner.del_item(elem)?;
        }
        Ok(())
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;

        if !&slf.is_truthy()? {
            Ok(format!("{}()", name))
        } else {
            let d_repr = slf
                .get()
                .most_common(py, None)
                // dict() preserves the ordering returned by most_common()
                .and_then(|x| x.as_any().pipe(PyDict::from_sequence))
                .or_else(|err| {
                    if err.is_instance_of::<PyTypeError>(py) {
                        // handle case where values are not orderable
                        slf.get().inner().clone_ref(py).into_bound(py).pipe(Ok)
                    } else {
                        Err(err)
                    }
                })?
                .repr()?;
            Ok(format!("{}({})", name, d_repr))
        }
    }

    fn __add__<'py>(&self, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let inner = self.inner_bind(py);
        let o = other.get();
        let result = PyDict::new(py);
        for (elem, count) in inner.iter() {
            let newcount = count.add(o.__getitem__(&elem)?)?;
            if newcount.gt(0)? {
                result.set_item(elem, newcount)?;
            }
        }
        for (elem, count) in o.inner_bind(py).iter() {
            if !inner.contains(&elem)? && count.gt(0)? {
                result.set_item(elem, count)?;
            }
        }
        Self::from_ref(result)
    }

    fn __sub__<'py>(&self, py: Python<'py>, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let inner = self.inner_bind(py);
        let o = other.get();
        let result = PyDict::new(py);
        for (elem, count) in inner.iter() {
            let newcount = count.sub(o.__getitem__(&elem)?)?;
            if newcount.gt(0)? {
                result.set_item(elem, newcount)?;
            }
        }
        for (elem, count) in o.inner_bind(py).iter() {
            if !inner.contains(&elem)? && count.lt(0)? {
                result.set_item(elem, PyInt::new(py, 0).sub(count)?)?;
            }
        }
        Self::from_ref(result)
    }

    fn __or__<'py>(&self, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let result = PyDict::new(py);
        let o = other.get();
        let inner = self.inner_bind(py);
        for (elem, count) in inner.iter() {
            let other_count = o.__getitem__(&elem)?;
            let newcount = pylibs::builtins::max_of(py, (&count, &other_count))?;
            if newcount.gt(0)? {
                result.set_item(elem, newcount)?
            }
        }
        for (elem, count) in other.get().inner_bind(other.py()).iter() {
            if !inner.contains(&elem)? && count.gt(0)? {
                result.set_item(elem, count)?
            }
        }
        Self::from_ref(result)
    }

    fn __and__<'py>(&self, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let o = other.get();
        let result = PyDict::new(py);
        for (elem, count) in self.inner_bind(py).iter() {
            let other_count = o.__getitem__(&elem)?;
            let newcount = pylibs::builtins::min_of(py, (&other_count, &count))?;
            if newcount.gt(0)? {
                result.set_item(elem, newcount)?;
            }
        }
        Self::from_ref(result)
    }

    fn __pos__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let result = PyDict::new(py);
        for (elem, count) in self.inner_bind(py).iter() {
            if count.gt(0)? {
                result.set_item(elem, count)?
            }
        }
        Self::from_ref(result)
    }

    fn __neg__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let result = PyDict::new(py);
        for (elem, count) in self.inner_bind(py).iter() {
            if count.lt(0)? {
                result.set_item(elem, count.neg()?)?;
            }
        }
        Self::from_ref(result)
    }

    fn __iadd__(&self, other: Bound<'_, PySupportsItems>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner_bind(py);
        for tup in other.items()?.try_iter()?.map(extract_tup_from_item) {
            let (elem, count) = tup?;
            let new_count = self.__getitem__(&elem)?.add(count)?;
            inner.set_item(elem, new_count)?;
        }
        keep_positive(&inner)
    }

    fn __isub__(&self, other: Bound<'_, PySupportsItems>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner_bind(py);
        for tup in other.items()?.try_iter()?.map(extract_tup_from_item) {
            let (elem, count) = tup?;
            let new_count = self.__getitem__(&elem)?.sub(count)?;
            inner.set_item(elem, new_count)?;
        }
        keep_positive(&inner)
    }

    fn __ior__(&self, other: Bound<'_, PySupportsItems>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner_bind(py);
        for tup in other.items()?.try_iter()?.map(extract_tup_from_item) {
            let (elem, other_count) = tup?;
            let count = self.__getitem__(&elem)?;
            if other_count.gt(count)? {
                inner.set_item(elem, other_count)?;
            }
        }
        keep_positive(&inner)
    }

    fn __iand__(&self, other: Bound<'_, PyMapping>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner_bind(py);
        for (elem, count) in inner.iter() {
            let other_count = other.as_any().get_item(&elem)?;
            if other_count.lt(count)? {
                inner.set_item(elem, other_count)?;
            }
        }
        keep_positive(&inner)
    }

    fn __ixor__<'py>(&self, other: Bound<'py, Self>) -> PyResult<()> {
        let py = other.py();
        let o = other.get();
        let inner = self.inner_bind(py);
        for (elem, count) in inner.iter() {
            let new_item = pylibs::builtins::abs(&count.sub(o.__getitem__(&elem)?)?)?;
            inner.set_item(elem, new_item)?
        }
        for (elem, count) in other.get().inner_bind(py).iter() {
            if !inner.contains(&elem)? {
                inner.set_item(elem, pylibs::builtins::abs(&count)?)?
            }
        }
        keep_positive(&inner)
    }
    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        let py = other.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match other {
                PyoCounter => {
                    let o = other.get();
                    for c in [self, o] {
                        for elem in c.inner_bind(py).try_iter().unwrap() {
                            let e = elem?;
                            if self.__getitem__(&e)?.eq(o.__getitem__(&e)?)? {
                                continue;
                            } else {
                                return Ok(false).map(Either::Left);
                            }
                        }
                    }
                    Ok(true).map(Either::Left)

                },
                PyDict => inner.eq(other).map(Either::Left),
                _ => Ok(PyNotImplemented::get(py).into_bound()).map(Either::Right),
            }
        }
    }

    fn __ne__<'py>(&self, other: &Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        if !other.is_instance_of::<PyoCounter>() {
            PyNotImplemented::get(other.py())
                .into_bound()
                .pipe(Ok)
                .map(Either::Right)
        } else {
            self.__eq__(other)?.map_left(|x| !x).pipe(Ok)
        }
    }

    fn __le__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        let py = other.py();
        let o = other.get();
        for c in [self, o] {
            for elem in c.inner_bind(py).try_iter().unwrap() {
                let e = elem?;
                if self.__getitem__(&e)?.le(o.__getitem__(&e)?)? {
                    continue;
                } else {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn __lt__<'py>(&self, other: &Bound<'py, Self>) -> PyCmpOut<bool, 'py> {
        let is_ne = self.__ne__(other)?;
        let is_le = self.__le__(other)?;
        is_ne.map_left(|is_ne| is_le && is_ne).pipe(Ok)
    }

    fn __ge__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        let py = other.py();
        let o = other.get();
        for c in [self, o] {
            for elem in c.inner_bind(py).try_iter().unwrap() {
                let e = elem?;
                if self.__getitem__(&e)?.ge(o.__getitem__(&e)?)? {
                    continue;
                } else {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    fn __gt__<'py>(&self, other: &Bound<'py, Self>) -> PyCmpOut<bool, 'py> {
        let is_ge = self.__ge__(other)?;
        self.__ne__(&other)?
            .map_left(|is_ne| is_ge && is_ne)
            .pipe(Ok)
    }

    fn __xor__<'py>(&self, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let inner = self.inner_bind(py);
        let o = other.get();
        let result = PyDict::new(py);
        for (elem, count) in inner.iter() {
            let newcount = pylibs::builtins::abs(&count.sub(o.__getitem__(&elem)?)?)?;
            if newcount.is_truthy()? {
                result.set_item(elem, newcount)?
            }
        }
        for (elem, count) in other.get().inner_bind(py).iter() {
            if !inner.contains(&elem)? && count.is_truthy()? {
                result.set_item(elem, pylibs::builtins::abs(&count)?)?
            }
        }
        Self::from_ref(result)
    }
}
impl ImplPyoReversible for PyoCounter {
    fn rev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>> {
        self.inner_bind(py)
            .as_any()
            .pipe(pylibs::builtins::reversed)
            .pipe(iterators::Iter::new)
    }
}
#[inline(always)]
fn extract_tup_from_item(
    x: PyResult<Bound<'_, PyAny>>,
) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyAny>)> {
    x?.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
}
#[inline]
fn update_counter(
    inner: &Bound<'_, PyDict>,
    iterable: Option<IntoUpdate<'_>>,
    kwargs: Option<Kwargs<'_>>,
) -> PyResult<()> {
    let py = inner.py();
    let zero = PyInt::new(py, 0).into_any();
    iterable
        .map(|iterable| match iterable {
            IntoUpdate::Dict(dict) => update_dict(inner, &dict, &zero),
            IntoUpdate::Mapping(mapping) => {
                if !inner.is_empty() {
                    for tup in mapping.items()?.try_iter()?.map(extract_tup_from_item) {
                        let (elem, count) = tup?;
                        let new_item =
                            count.add(inner.get_item(&elem)?.unwrap_or_else(|| zero.to_owned()))?;
                        inner.set_item(elem, new_item)?
                    }
                } else {
                    // fast path when counter is empty
                    inner.update(&mapping)?;
                }

                Ok(())
            }
            IntoUpdate::Iterable(it) => {
                for elem in it.try_iter()? {
                    let e = elem?;
                    let new_item = inner
                        .get_item(&e)?
                        .unwrap_or_else(|| zero.to_owned())
                        .add(1)?;
                    inner.set_item(&e, new_item)?;
                }

                Ok(())
            }
        })
        .transpose()?;

    kwargs
        .map(|kw| update_dict(inner, &kw, &zero))
        .transpose()?;
    Ok(())
}
#[inline]
fn update_dict(
    inner: &Bound<'_, PyDict>,
    dict: &Bound<'_, PyDict>,
    zero: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if !inner.is_empty() {
        for (elem, count) in dict.iter() {
            let new_item = count.add(inner.get_item(&elem)?.unwrap_or_else(|| zero.to_owned()))?;
            inner.set_item(elem, new_item)?
        }
        Ok(())
    } else {
        // fast path when counter is empty
        inner.update(dict.as_mapping())
    }
}
#[inline]
fn subtract_counter(
    inner: &Bound<'_, PyDict>,
    iterable: Option<IntoUpdate<'_>>,
    kwargs: Option<Kwargs<'_>>,
) -> PyResult<()> {
    let py = inner.py();
    let zero = PyInt::new(py, 0).into_any();
    iterable
        .map(|it| match it {
            IntoUpdate::Dict(dict) => subtract_dict(inner, &dict, &zero),
            IntoUpdate::Mapping(mapping) => {
                for tup in mapping.items()?.try_iter()?.map(extract_tup_from_item) {
                    let (elem, count) = tup?;
                    let new_item = inner
                        .get_item(&elem)?
                        .unwrap_or_else(|| zero.to_owned())
                        .sub(count)?;
                    inner.set_item(elem, new_item)?
                }

                Ok(())
            }
            IntoUpdate::Iterable(it) => {
                for elem in it.try_iter()? {
                    let e = elem?;
                    let new_item = inner
                        .get_item(&e)?
                        .unwrap_or_else(|| zero.to_owned())
                        .sub(1)?;
                    inner.set_item(&e, new_item)?
                }
                Ok(())
            }
        })
        .transpose()?;
    kwargs
        .map(|dict| subtract_dict(inner, &dict, &zero))
        .transpose()?;
    Ok(())
}

#[inline]
fn subtract_dict(
    inner: &Bound<'_, PyDict>,
    dict: &Bound<'_, PyDict>,
    zero: &Bound<'_, PyAny>,
) -> PyResult<()> {
    for (elem, count) in dict.iter() {
        let new_item = inner
            .get_item(&elem)?
            .unwrap_or_else(|| zero.to_owned())
            .sub(count)?;
        inner.set_item(elem, new_item)?
    }
    Ok(())
}
fn keep_positive(inner: &Bound<'_, PyDict>) -> PyResult<()> {
    inner
        .items_view()
        .try_iter()
        .unwrap()
        .map(|x| x?.extract::<(Bound<'_, PyAny>, isize)>())
        .filter_map(|kv| match kv {
            Ok((elem, count)) => {
                if !(count > 0) {
                    Some(Ok(elem))
                } else {
                    None
                }
            }
            Err(err) => Some(Err(err)),
        })
        .collect::<PyResult<Vec<_>>>()?
        .iter()
        .try_for_each(|elem| inner.del_item(elem))
}
