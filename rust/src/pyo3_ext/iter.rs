use pyo3::{
    IntoPyObjectExt, ffi,
    prelude::*,
    types::{PyFrozenSet, PyFrozenSetBuilder, PyList, PySet},
};

/// Trait for Python collection types that can be created from an iterator of `Bound<'_, PYAny>`.
/// Much more flexible than Pyo3 provided creations, who often require `ExactSizedIterator`, which is quickly limiting.
pub trait TryFromBoundIterator<'py, T: IntoPyObject<'py>>: Sized {
    fn try_from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = PyResult<T>>;
}
pub trait FromBoundIterator<'py, T>: Sized
where
    T: IntoPyObject<'py>,
{
    fn from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = T>;
}
/// Mirror of std `FromIterator` trait, but for PyO3 `Bound<'_, PyIterator>` instead of std `Iterator`.
pub trait TryCollectBoundIterator<'py, T: IntoPyObject<'py>>:
    IntoIterator<Item = PyResult<T>> + Sized
{
    #[inline(always)]
    fn try_collect_bound<B>(self, py: Python<'py>) -> PyResult<Bound<'py, B>>
    where
        B: TryFromBoundIterator<'py, T>,
    {
        B::try_from_iter_bound(self, py)
    }
}

impl<'py, I> TryCollectBoundIterator<'py, Bound<'py, PyAny>> for I where
    I: IntoIterator<Item = PyResult<Bound<'py, PyAny>>>
{
}

pub trait CollectBoundIterator<'py>: IntoIterator + Sized
where
    Self::Item: IntoPyObject<'py>,
{
    #[inline(always)]
    fn collect_bound<B>(self, py: Python<'py>) -> PyResult<Bound<'py, B>>
    where
        B: FromBoundIterator<'py, Self::Item>,
    {
        B::from_iter_bound(self, py)
    }
}

impl<'py, I> CollectBoundIterator<'py> for I
where
    I: IntoIterator,
    I::Item: IntoPyObject<'py>,
{
}
impl<'py, T: IntoPyObject<'py>> FromBoundIterator<'py, T> for PyList {
    #[inline(always)]
    fn from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = T>,
    {
        PyList::new(py, iter)
    }
}
impl<'py, T: IntoPyObject<'py>> TryFromBoundIterator<'py, T> for PyFrozenSet {
    fn try_from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = PyResult<T>>,
    {
        let mut builder = PyFrozenSetBuilder::new(py)?;
        iter.into_iter()
            .try_for_each(|item| builder.add(item?))
            .map(|_| builder.finalize())
    }
}
impl<'py, T: IntoPyObject<'py>> TryFromBoundIterator<'py, T> for PySet {
    fn try_from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = PyResult<T>>,
    {
        let pyset = PySet::empty(py)?;
        iter.into_iter()
            .try_for_each(|item| pyset.add(item?))
            .map(|_| pyset)
    }
}
impl<'py, T: IntoPyObject<'py>> TryFromBoundIterator<'py, T> for PyList {
    fn try_from_iter_bound<I: IntoIterator<Item = PyResult<T>>>(
        iter: I,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, Self>> {
        let mut elements = iter.into_iter();
        let (min_len, _) = elements.size_hint();
        // PyList_New checks for overflow but has a bad error message, so we check ourselves
        let len: ffi::Py_ssize_t = min_len
            .try_into()
            .expect("out of range integral type conversion attempted on `elements.len()`");

        let list = unsafe {
            Bound::from_owned_ptr(py, ffi::PyList_New(len)).cast_into_unchecked::<PyList>()
        };

        (&mut elements)
            .take(min_len)
            .enumerate()
            .try_for_each(|(count, item)| unsafe {
                ffi::PyList_SET_ITEM(
                    list.as_ptr(),
                    count as ffi::Py_ssize_t,
                    item?.into_bound_py_any(py)?.into_ptr(),
                );
                Ok::<(), PyErr>(())
            })?;

        elements.try_for_each(|item| list.append(item?))?;

        Ok(list)
    }
}
