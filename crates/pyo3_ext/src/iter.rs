use pyo3::{
    IntoPyObjectExt, ffi,
    prelude::*,
    types::{PyFrozenSet, PyFrozenSetBuilder, PyList, PySet, PyTuple},
};

/// Trait for Python collection types that can be created from an iterator of `Bound<'_, PYAny>`.
/// Much more flexible than Pyo3 provided creations, who often require `ExactSizedIterator`, which is quickly limiting.
pub trait TryFromBoundIterator<'py, I>: Sized
where
    I: IntoIterator<Item = PyResult<Self::Item>>,
{
    type Item: IntoPyObject<'py>;

    fn try_from_iter_bound(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>;
}
pub trait FromBoundIterator<'py, I>: Sized
where
    I: IntoIterator,
    I::Item: IntoPyObject<'py>,
{
    fn from_iter_bound(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>;
}
/// Mirror of std `Iterator::collect`, but for `PyO3` Python collection types.
pub trait CollectBoundIterator<'py>: Iterator + Sized {
    #[inline(always)]
    fn collect_bound<B>(self, py: Python<'py>) -> PyResult<Bound<'py, B>>
    where
        Self::Item: IntoPyObject<'py>,
        B: FromBoundIterator<'py, Self>,
    {
        B::from_iter_bound(self, py)
    }

    #[inline(always)]
    fn try_collect_bound<B>(self, py: Python<'py>) -> PyResult<Bound<'py, B>>
    where
        B: TryFromBoundIterator<'py, Self>,
        Self: Iterator<Item = PyResult<B::Item>>,
    {
        B::try_from_iter_bound(self, py)
    }
}
impl<I> CollectBoundIterator<'_> for I where I: Iterator {}
impl<'py, T, I> FromBoundIterator<'py, I> for PyTuple
where
    T: IntoPyObject<'py>,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator<Item = T>,
{
    #[inline(always)]
    fn from_iter_bound(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        PyTuple::new(py, iter)
    }
}
macro_rules! impl_from_bound_iterator {
    ($target:ty) => {
        impl<'py, I> FromBoundIterator<'py, I> for $target
        where
            I: IntoIterator,
            I::Item: IntoPyObject<'py>,
        {
            #[inline(always)]
            fn from_iter_bound(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
                Self::new(py, iter)
            }
        }
    };
}

impl_from_bound_iterator!(PyList);
impl_from_bound_iterator!(PySet);
impl<'py, I> FromBoundIterator<'py, I> for PyFrozenSet
where
    I: IntoIterator,
    I::Item: IntoPyObject<'py>,
{
    #[inline(always)]
    fn from_iter_bound(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let mut builder = PyFrozenSetBuilder::new(py)?;
        iter.into_iter()
            .try_for_each(|item| builder.add(item))
            .map(|()| builder.finalize())
    }
}
impl<'py, T, I> TryFromBoundIterator<'py, I> for PyFrozenSet
where
    T: IntoPyObject<'py>,
    I: IntoIterator<Item = PyResult<T>>,
{
    type Item = T;

    #[inline(always)]
    fn try_from_iter_bound(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let mut builder = PyFrozenSetBuilder::new(py)?;
        iter.into_iter()
            .try_for_each(|item| builder.add(item?))
            .map(|()| builder.finalize())
    }
}
impl<'py, T, I> TryFromBoundIterator<'py, I> for PySet
where
    T: IntoPyObject<'py>,
    I: IntoIterator<Item = PyResult<T>>,
{
    type Item = T;

    #[inline(always)]
    fn try_from_iter_bound(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let pyset = PySet::empty(py)?;
        iter.into_iter()
            .try_for_each(|item| pyset.add(item?))
            .map(|()| pyset)
    }
}

impl<'py, T, I> TryFromBoundIterator<'py, I> for PyList
where
    T: IntoPyObject<'py>,
    I: IntoIterator<Item = PyResult<T>>,
{
    type Item = T;
    #[allow(clippy::cast_possible_wrap)]
    #[inline(always)]
    fn try_from_iter_bound(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
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
