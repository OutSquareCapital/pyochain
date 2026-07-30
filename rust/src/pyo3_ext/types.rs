use crate::pyobject_native_type_named;
use either::Either;
use pyo3::{
    PyTypeInfo,
    call::PyCallArgs,
    exceptions::PyTypeError,
    ffi::{self, PyDictValues, PyTypeObject},
    intern,
    prelude::*,
    sync::PyOnceLock,
    types::{
        PyBool, PyDict, PyDictItems, PyDictKeys, PyFrozenSet, PyFrozenSetBuilder, PyInt,
        PyIterator, PyList, PyNotImplemented, PyRange, PySequence, PySet, PySlice, PyTuple, PyType,
    },
};
use tap::prelude::*;
const COLLECTIONS_ABC: &str = "collections.abc";
/// Return type from python comparison dunders, returning either `T` in case of success, or `NotImplemented`.
pub type PyCmpOut<'py, T> = PyResult<Either<T, Bound<'py, PyNotImplemented>>>;
/// All ABCs have a `register` method that can be used to register a type as a virtual subclass of the ABC.\
/// This trait factorize the implementation for all ABCs.\
/// The code is strictly identical from what's already available for `pyo3::types::PySequence` for example.
pub trait ABCRegister<'py>: PyTypeInfo {
    fn register<T: PyTypeInfo>(py: Python<'_>) -> PyResult<()> {
        let ty = T::type_object(py);
        Self::type_object(py).call_method1("register", (ty,))?;
        Ok(())
    }
}

impl ABCRegister<'_> for PyMutableSequence {}
impl ABCRegister<'_> for PyAbstractSet {}
impl ABCRegister<'_> for PyIterable {}
impl ABCRegister<'_> for PyMutableSet {}
impl ABCRegister<'_> for PyIterator {}
/// Trait for Python collection types that can be created from an iterator of `Bound<'_, PYAny>`.
/// Much more flexible than Pyo3 provided creations, who often require `ExactSizedIterator`, which is quickly limiting.
pub trait FromBoundIterator<'py>: Sized {
    fn from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = PyResult<Bound<'py, PyAny>>>;
}

impl<'py> FromBoundIterator<'py> for PyFrozenSet {
    fn from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = PyResult<Bound<'py, PyAny>>>,
    {
        let mut builder = PyFrozenSetBuilder::new(py)?;
        iter.into_iter()
            .try_for_each(|item| builder.add(item?))
            .map(|_| builder.finalize())
    }
}
impl<'py> FromBoundIterator<'py> for PySet {
    fn from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = PyResult<Bound<'py, PyAny>>>,
    {
        let pyset = PySet::empty(py)?;
        iter.into_iter()
            .try_for_each(|item| pyset.add(item?))
            .map(|_| pyset)
    }
}
impl<'py> FromBoundIterator<'py> for PyList {
    fn from_iter_bound<I>(iter: I, py: Python<'py>) -> PyResult<Bound<'py, Self>>
    where
        I: IntoIterator<Item = PyResult<Bound<'py, PyAny>>>,
    {
        let mut iter = iter.into_iter();
        let (min_len, _) = iter.size_hint();

        let len: ffi::Py_ssize_t = min_len
            .try_into()
            .expect("out of range integral type conversion attempted on iterator size_hint");

        let list = unsafe {
            Bound::from_owned_ptr(py, ffi::PyList_New(len)).cast_into_unchecked::<PyList>()
        };

        let count = (&mut iter)
            .take(min_len)
            .try_fold(0, |count, item| unsafe {
                ffi::PyList_SET_ITEM(list.as_ptr(), count as ffi::Py_ssize_t, item?.into_ptr());
                Ok::<_, PyErr>(count + 1)
            })?;

        debug_assert_eq!(
            count, min_len,
            "iterator produced fewer than its size_hint lower bound"
        );

        iter.try_for_each(|item| list.append(item?))?;

        Ok(list)
    }
}
/// Mirror of std `FromIterator` trait, but for PyO3 `Bound<'_, PyIterator>` instead of std `Iterator`.
pub trait CollectBoundIterator<'py>:
    IntoIterator<Item = PyResult<Bound<'py, PyAny>>> + Sized
{
    fn collect_bound<B>(self, py: Python<'py>) -> PyResult<Bound<'py, B>>
    where
        B: FromBoundIterator<'py>,
    {
        B::from_iter_bound(self, py)
    }
}

impl<'py, I> CollectBoundIterator<'py> for I where
    I: IntoIterator<Item = PyResult<Bound<'py, PyAny>>>
{
}
pub trait PySequenceExtMethods<'py> {
    fn count(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>>;

    fn index(
        &self,
        value: &Bound<'py, PyAny>,
        start: Option<&Bound<'py, PyAny>>,
        stop: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>>;
}
macro_rules! impl_sequence_ext_methods {
    ($($t:ty),*) => {
        $(
            impl<'py> PySequenceExtMethods<'py> for Bound<'py, $t> {
                fn count(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
                    self.call_method1(intern!(self.py(), "count"), (value,))
                        .map(|x| unsafe { x.cast_into_unchecked::<PyInt>() })
                }

                fn index(
                    &self,
                    value: &Bound<'py, PyAny>,
                    start: Option<&Bound<'py, PyAny>>,
                    stop: Option<&Bound<'py, PyAny>>,
                ) -> PyResult<Bound<'py, PyAny>> {
                    let method_name = intern!(self.py(), "index");
                    match (start, stop) {
                        (Some(start), Some(stop)) => self.call_method1(method_name, (value, start, stop)),
                        (Some(start), None) => self.call_method1(method_name, (value, start)),
                        (None, Some(stop)) => self.call_method1(method_name, (value, stop)),
                        (None, None) => self.call_method1(method_name, (value,)),
                    }
                }
            }
        )*
    };
}

impl_sequence_ext_methods!(PyList, PyTuple, PyDeque);
/// The `index` method is different on `range`, so we need to implement it separately.
pub trait PyRangeExtMethods<'py> {
    fn count(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>>;

    fn index(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>>;
}
impl<'py> PyRangeExtMethods<'py> for Bound<'py, PyRange> {
    fn count(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.call_method1(intern!(self.py(), "count"), (value,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyInt>() })
    }
    fn index(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.call_method1(intern!(self.py(), "index"), (value,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyInt>() })
    }
}
pub trait PySetExtMethods<'py>: Sized {
    fn difference<O: PyCallArgs<'py>>(&self, others: O) -> PyResult<Self>;
    fn isdisjoint(&self, s: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>>;
    fn issubset(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>>;
    fn issuperset(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>>;
    fn intersection<O: PyCallArgs<'py>>(&self, s: O) -> PyResult<Self>;
    fn symmetric_difference(&self, other: Bound<'py, PyAny>) -> PyResult<Self>;
    fn union<O: PyCallArgs<'py>>(&self, others: O) -> PyResult<Self>;
}
pub trait PySetExtMethodsMut<'py>: PySetExtMethods<'py> {
    fn difference_update<O: PyCallArgs<'py>>(&self, s: O) -> PyResult<()>;
    fn intersection_update<O: PyCallArgs<'py>>(&self, s: O) -> PyResult<()>;
    fn remove(&self, element: Bound<'_, PyAny>) -> PyResult<()>;
    fn symmetric_difference_update(&self, s: Bound<'_, PyAny>) -> PyResult<()>;
    fn update<O: PyCallArgs<'py>>(&self, s: O) -> PyResult<()>;
}
impl<'py> PySetExtMethodsMut<'py> for Bound<'py, PySet> {
    fn difference_update<O: PyCallArgs<'py>>(&self, s: O) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "difference_update"), s)?;
        Ok(())
    }
    fn intersection_update<O: PyCallArgs<'py>>(&self, s: O) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "intersection_update"), s)?;
        Ok(())
    }
    fn remove(&self, element: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(element.py(), "remove"), (element,))?;
        Ok(())
    }
    fn symmetric_difference_update(&self, s: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(s.py(), "symmetric_difference_update"), (s,))?;
        Ok(())
    }
    fn update<O: PyCallArgs<'py>>(&self, s: O) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "update"), s)?;
        Ok(())
    }
}
macro_rules! impl_sequence_ext_methods {
    ($($t:ty),*) => {
        $(
            impl<'py> PySetExtMethods<'py> for Bound<'py, $t> {
                fn isdisjoint(&self, s: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
                    self.call_method1(intern!(self.py(), "isdisjoint"), (s,))
                        .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
                }

                fn issubset(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
                    self.call_method1(intern!(self.py(), "issubset"), (other,))
                        .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
                }

                fn issuperset(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
                    self.call_method1(intern!(self.py(), "issuperset"), (other,))
                        .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
                }

                fn intersection<O: PyCallArgs<'py>>(&self, s: O) -> PyResult<Self> {
                    self.call_method1(intern!(self.py(), "intersection"), s)
                        .map(|x| unsafe { x.cast_into_unchecked::<$t>() })
                }
                fn union<O: PyCallArgs<'py>>(&self, others: O) -> PyResult<Self> {
                    self.call_method1(intern!(self.py(), "union"), others)
                        .map(|x| unsafe { x.cast_into_unchecked::<$t>() })
                }
                fn difference<O: PyCallArgs<'py>>(&self, others: O) -> PyResult<Self> {
                    self.call_method1(intern!(self.py(), "difference"), others)
                        .map(|x| unsafe { x.cast_into_unchecked::<$t>() })
                }

                fn symmetric_difference(&self, other: Bound<'py, PyAny>) -> PyResult<Self> {
                    self.call_method1(intern!(self.py(), "symmetric_difference"), (other,))
                        .map(|x| unsafe { x.cast_into_unchecked::<$t>() })
                }
            }
        )*
    };
}
impl_sequence_ext_methods!(PySet, PyFrozenSet);
pub trait PyListExtMethods<'py> {
    fn clear(&self) -> ();
    fn extend(&self, iterable: Bound<'_, PyAny>) -> PyResult<()>;
    fn last(&self) -> PyResult<Bound<'py, PyAny>>;
    fn sort_by(&self, key: &Bound<'_, PyAny>, reverse: bool) -> PyResult<()>;
}
impl<'py> PyListExtMethods<'py> for Bound<'py, PyList> {
    fn clear(&self) -> () {
        unsafe { ffi::PyList_Clear(self.as_ptr()) };
    }

    fn extend(&self, iterable: Bound<'_, PyAny>) -> PyResult<()> {
        iterable
            .try_iter()
            .map(|_| unsafe { ffi::PyList_Extend(self.as_ptr(), iterable.as_ptr()) })?;
        Ok(())
    }
    fn last(&self) -> PyResult<Bound<'py, PyAny>> {
        self.as_any().get_item(-1)
    }

    fn sort_by(&self, key: &Bound<'_, PyAny>, reverse: bool) -> PyResult<()> {
        let py = self.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "key"), key)?;
        kwargs.set_item(intern!(py, "reverse"), reverse)?;
        self.call_method(intern!(py, "sort"), (), Some(&kwargs))?;
        Ok(())
    }
}

/// Type representing the `typing.SupportsIndex` protocol.
#[repr(transparent)]
pub struct PySupportsIndex(PyAny);
pyobject_native_type_named!(PySupportsIndex);
unsafe impl PyTypeInfo for PySupportsIndex {
    const NAME: &'static str = "SupportsIndex";
    const MODULE: Option<&'static str> = Some("typing");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, "typing", "SupportsIndex")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        let py = object.py();
        object
            .hasattr(intern!(py, "__index__"))
            .unwrap_or_else(|err| {
                err.write_unraisable(object.py(), Some(object));
                false
            })
            || object
                .is_instance(&Self::type_object(py).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(py, Some(object));
                    false
                })
    }
}
pub trait PySupportsIndexMethods<'py> {
    fn index(&self) -> PyResult<Bound<'py, PyInt>>;
}
impl<'py> PySupportsIndexMethods<'py> for Bound<'py, PySupportsIndex> {
    fn index(&self) -> PyResult<Bound<'py, PyInt>> {
        self.call_method0(intern!(self.py(), "__index__"))
            .map(|x| unsafe { x.cast_into_unchecked::<PyInt>() })
    }
}
/// Type representing the `collections.abc.Iterable` abstract base class.
#[repr(transparent)]
pub struct PyIterable(PyAny);
pyobject_native_type_named!(PyIterable);
unsafe impl PyTypeInfo for PyIterable {
    const NAME: &'static str = "Iterable";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "Iterable")
            .unwrap()
            .as_type_ptr()
    }
    // NOTE: Even if `PySequence_Check` won't return always return `True` on `isinstance(x, Iterable)` (e.g for `__getitem__`-only objects),\
    // not using it here would mean runtime errors on objects that are perfectly valid when calling `try_iter()` on them.
    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        unsafe { (*(*object.as_ptr()).ob_type).tp_iter }.is_some()
            || unsafe { ffi::PySequence_Check(object.as_ptr()) != 0 }
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(object.py(), Some(object));
                    false
                })
    }
}
/// Type representing the `collections.abc.MutableSequence` abstract base class.
#[repr(transparent)]
pub struct PyMutableSequence(PyAny);
pyobject_native_type_named!(PyMutableSequence);
unsafe impl PyTypeInfo for PyMutableSequence {
    const NAME: &'static str = "MutableSequence";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "MutableSequence")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PyList::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(object.py(), Some(object));
                    false
                })
    }
}
pub trait PyMutableSequenceMethods<'py> {
    fn set_slice_with_step(
        &self,
        start: isize,
        stop: isize,
        step: isize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()>;
}
impl<'py> PyMutableSequenceMethods<'py> for Bound<'py, PyMutableSequence> {
    fn set_slice_with_step(
        &self,
        start: isize,
        stop: isize,
        step: isize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = value.py();

        let slice = PySlice::new(py, start, stop, step);
        self.set_item(&slice, value)
    }
}
/// Type representing the `collections.abc.Set` abstract base class.
#[repr(transparent)]
pub struct PyAbstractSet(PyAny);
pyobject_native_type_named!(PyAbstractSet);
unsafe impl PyTypeInfo for PyAbstractSet {
    const NAME: &'static str = "Set";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "Set")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PySet::is_type_of(object)
            || PyFrozenSet::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(object.py(), Some(object));
                    false
                })
    }
}
/// Type representing the `collections.abc.Set` abstract base class.
#[repr(transparent)]
pub struct PyMutableSet(PyAny);
pyobject_native_type_named!(PyMutableSet);
unsafe impl PyTypeInfo for PyMutableSet {
    const NAME: &'static str = "MutableSet";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "MutableSet")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PySet::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(object.py(), Some(object));
                    false
                })
    }
}

pub trait PyMutableSetMethods<'py> {
    fn add(&self, value: &Bound<'_, PyAny>) -> PyResult<()>;
    fn discard(&self, value: &Bound<'_, PyAny>) -> PyResult<()>;
}
impl PyMutableSetMethods<'_> for Bound<'_, PyMutableSet> {
    fn add(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "add"), (value,))?;
        Ok(())
    }
    fn discard(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "discard"), (value,))?;
        Ok(())
    }
}
/// Type representing the `collections.deque` class.
#[repr(transparent)]
pub struct PyDeque(PyAny);
pyobject_native_type_named!(PyDeque);
unsafe impl PyTypeInfo for PyDeque {
    const NAME: &'static str = "deque";
    const MODULE: Option<&'static str> = Some("collections");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, "collections", "deque")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        object
            .is_instance(&Self::type_object(object.py()).into_any())
            .unwrap_or_else(|err| {
                err.write_unraisable(object.py(), Some(object));
                false
            })
    }
}

impl PyDeque {
    #[inline]
    pub fn new<'py>(
        py: Python<'py>,
        iterable: Bound<'py, PyAny>,
        maxlen: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::type_object(py)
            .call1((iterable, maxlen))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
}
pub trait PyDequeMethods<'py> {
    /// Returns `self` cast as a `PySequence`.
    fn as_sequence(&self) -> &Bound<'py, PySequence>;
    fn append(&self, x: Bound<'_, PyAny>) -> PyResult<()>;
    fn append_left(&self, x: Bound<'_, PyAny>) -> PyResult<()>;
    fn extend(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()>;
    fn clear(&self) -> PyResult<()>;
    fn extend_left(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()>;
    fn rotate(&self, n: isize) -> PyResult<()>;
    fn insert(&self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn pop(&self) -> PyResult<Bound<'py, PyAny>>;
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn reversed(&self) -> PyResult<Bound<'py, PyIterator>>;
}
impl<'py> PyDequeMethods<'py> for Bound<'py, PyDeque> {
    /// Returns `self` cast as a `PySequence`.
    fn as_sequence(&self) -> &Bound<'py, PySequence> {
        unsafe { self.cast_unchecked() }
    }

    fn append(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "append"), (x,))?;
        Ok(())
    }

    fn append_left(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "appendleft"), (x,))?;
        Ok(())
    }

    fn extend(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "extend"), (iterable,))?;
        Ok(())
    }

    fn clear(&self) -> PyResult<()> {
        self.call_method0(intern!(self.py(), "clear"))?;
        Ok(())
    }

    fn extend_left(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "extendleft"), (iterable,))?;
        Ok(())
    }

    fn rotate(&self, n: isize) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "rotate"), (n,))?;
        Ok(())
    }

    fn insert(&self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "insert"), (index, value))?;
        Ok(())
    }
    fn pop(&self) -> PyResult<Bound<'py, PyAny>> {
        self.call_method0(intern!(self.py(), "pop"))
    }
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(value.py(), "remove"), (value,))?;
        Ok(())
    }
    fn reversed(&self) -> PyResult<Bound<'py, PyIterator>> {
        self.call_method0(intern!(self.py(), "__reversed__"))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
    }
}
/// Type representing the `collections.abc.Set` abstract base class.
#[repr(transparent)]
pub struct PySupportsItems(PyAny);
pyobject_native_type_named!(PySupportsItems);
unsafe impl PyTypeInfo for PySupportsItems {
    const NAME: &'static str = "SupportsItems";
    const MODULE: Option<&'static str> = Some("");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "SupportsItems")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        object
            .hasattr(intern!(object.py(), "items"))
            .unwrap_or_else(|err| {
                err.write_unraisable(object.py(), Some(object));
                false
            })
    }
}

pub trait PySupportsItemsMethods<'py> {
    fn items(&self) -> PyResult<Bound<'py, PyAbstractSet>>;
}
impl<'py> PySupportsItemsMethods<'py> for Bound<'py, PySupportsItems> {
    fn items(&self) -> PyResult<Bound<'py, PyAbstractSet>> {
        self.call_method0(intern!(self.py(), "items"))
            .and_then(|x| {
                x.cast_into::<PyAbstractSet>()
                    .map_err(|_| PyTypeError::new_err(""))
            })
    }
}
#[allow(unused)]
pub trait IntoPyMappingView<'py> {
    fn items_view(&self) -> Bound<'py, PyDictItems>;
    fn keys_view(&self) -> Bound<'py, PyDictKeys>;
    fn values_view(&self) -> Bound<'py, PyDictValues>;
}
impl<'py> IntoPyMappingView<'py> for Bound<'py, PyDict> {
    fn items_view(&self) -> Bound<'py, PyDictItems> {
        self.call_method0(intern!(self.py(), "items"))
            .unwrap()
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyDictItems>() })
    }
    fn keys_view(&self) -> Bound<'py, PyDictKeys> {
        self.call_method0(intern!(self.py(), "keys"))
            .unwrap()
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyDictKeys>() })
    }
    fn values_view(&self) -> Bound<'py, PyDictValues> {
        self.call_method0(intern!(self.py(), "values"))
            .unwrap()
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyDictValues>() })
    }
}
pub mod pyitertools {
    use super::*;
    /// Type representing the `itertools.repeat` iterator.
    #[repr(transparent)]
    pub struct PyRepeat(PyAny);
    pyobject_native_type_named!(PyRepeat);
    unsafe impl PyTypeInfo for PyRepeat {
        const NAME: &'static str = "repeat";
        const MODULE: Option<&'static str> = Some("itertools");

        #[inline]
        fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
            static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
            TYPE.import(py, "itertools", "repeat")
                .unwrap()
                .as_type_ptr()
        }

        #[inline]
        fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
            object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(object.py(), Some(object));
                    false
                })
        }
    }
    impl PyRepeat {
        #[inline(always)]
        pub fn new<'py>(
            obj: &Bound<'py, PyAny>,
            n: Option<&Bound<'py, PyInt>>,
        ) -> PyResult<Bound<'py, PyIterator>> {
            let py = obj.py();
            Self::type_object(py)
                .pipe(|func| match n {
                    Some(n) => func.call1((obj, n)),
                    None => func.call1((obj,)),
                })
                .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
        }
    }
}
