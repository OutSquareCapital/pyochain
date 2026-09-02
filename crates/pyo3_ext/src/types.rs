use crate::pyobject_native_type_named;
use either::Either;
use pyo3::{
    BoundObject, PyTypeInfo,
    exceptions::{PyKeyError, PyTypeError},
    ffi, intern,
    prelude::*,
    sync::PyOnceLock,
    types::{
        PyDict, PyDictItems, PyDictKeys, PyDictValues, PyFrozenSet, PyInt, PyIterator, PyList,
        PyNotImplemented, PySequence, PySet, PySlice, PyType,
    },
};
use tap::prelude::*;
const COLLECTIONS_ABC: &str = "collections.abc";
/// Return type from python comparison dunders, returning either `T` in case of success, or `NotImplemented`.
pub type PyCmpOut<'py, T> = PyResult<Either<T, Bound<'py, PyNotImplemented>>>;
/// Small extension trait for `PyNotImplemented` to allow for easy conversion to `PyCmpOut`.
pub trait FromCmp<'py, T> {
    fn from_cmp(py: Python<'py>) -> PyCmpOut<'py, T>;
}

impl<'py, T> FromCmp<'py, T> for PyNotImplemented {
    fn from_cmp(py: Python<'py>) -> PyCmpOut<'py, T> {
        Ok(PyNotImplemented::get(py).into_bound().pipe(Either::Right))
    }
}
/// Output of a `pop` operation on a mutable data structure, e.g a `set` or a `dict`.
pub enum PopResult<'py> {
    /// Succes, code `1`.
    Ok(Bound<'py, PyAny>),
    /// Key missing, code `0`.
    KeyMissing,
    /// Error, code `-1`.
    Err(PyErr),
}
impl<'py> PopResult<'py> {
    /// Converts a `PopResult` into a `PyResult`.\
    /// `Ok` => `Ok`, `KeyMissing` => `Err(PyKeyError(""))`, `Err` => `Err`.
    pub fn into_pyresult(self) -> PyResult<Bound<'py, PyAny>> {
        match self {
            PopResult::Ok(v) => Ok(v),
            PopResult::Err(e) => Err(e),
            PopResult::KeyMissing => Err(PyKeyError::new_err("")),
        }
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
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
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
            .unwrap_or_else(|err| false_and_write(err, object))
            || object
                .is_instance(&Self::type_object(py).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
    }
}
#[repr(transparent)]
pub struct PyMappingView(PyAny);
pyobject_native_type_named!(PyMappingView);
unsafe impl PyTypeInfo for PyMappingView {
    const NAME: &'static str = "MappingView";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);
    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "MappingView")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        let py = object.py();
        PyDictItems::is_type_of(object)
            || PyDictKeys::is_type_of(object)
            || object
                .is_instance(&Self::type_object(py).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
    }
}

#[repr(transparent)]
pub struct PyMutableMapping(PyAny);
pyobject_native_type_named!(PyMutableMapping);
unsafe impl PyTypeInfo for PyMutableMapping {
    const NAME: &'static str = "MutableMapping";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);
    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "MutableMapping")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PyDict::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
    }
}

#[repr(transparent)]
pub struct PyKeysView(PyAny);
pyobject_native_type_named!(PyKeysView);
unsafe impl PyTypeInfo for PyKeysView {
    const NAME: &'static str = "KeysView";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);
    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "KeysView")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PyDictKeys::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
    }
}

#[repr(transparent)]
pub struct PyValuesView(PyAny);
pyobject_native_type_named!(PyValuesView);
unsafe impl PyTypeInfo for PyValuesView {
    const NAME: &'static str = "ValuesView";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);
    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "ValuesView")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PyDictValues::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
    }
}

#[repr(transparent)]
pub struct PyItemsView(PyAny);
pyobject_native_type_named!(PyItemsView);
unsafe impl PyTypeInfo for PyItemsView {
    const NAME: &'static str = "ItemsView";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);
    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "ItemsView")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PyDictItems::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
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
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
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
                .unwrap_or_else(|err| false_and_write(err, object))
    }
}
#[repr(transparent)]
pub struct PySized(PyAny);
pyobject_native_type_named!(PySized);
unsafe impl PyTypeInfo for PySized {
    const NAME: &'static str = "Sized";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "Sized")
            .unwrap()
            .as_type_ptr()
    }
    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        let py = object.py();
        object
            .hasattr(intern!(py, "__len__"))
            .unwrap_or_else(|err| false_and_write(err, object))
            || object
                .is_instance(&Self::type_object(py).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
    }
}
#[repr(transparent)]
pub struct PyContainer(PyAny);
pyobject_native_type_named!(PyContainer);
unsafe impl PyTypeInfo for PyContainer {
    const NAME: &'static str = "Container";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "Container")
            .unwrap()
            .as_type_ptr()
    }
    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        let py = object.py();
        object
            .hasattr(intern!(py, "__contains__"))
            .unwrap_or_else(|err| false_and_write(err, object))
            || object
                .is_instance(&Self::type_object(py).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
    }
}

#[repr(transparent)]
pub struct PyCollection(PyAny);
pyobject_native_type_named!(PyCollection);
unsafe impl PyTypeInfo for PyCollection {
    const NAME: &'static str = "Collection";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "Collection")
            .unwrap()
            .as_type_ptr()
    }
    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        object
            .is_instance(&Self::type_object(object.py()).into_any())
            .unwrap_or_else(|err| false_and_write(err, object))
    }
}

#[repr(transparent)]
pub struct PyReversible(PyAny);
pyobject_native_type_named!(PyReversible);
unsafe impl PyTypeInfo for PyReversible {
    const NAME: &'static str = "Reversible";
    const MODULE: Option<&'static str> = Some(COLLECTIONS_ABC);

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "Reversible")
            .unwrap()
            .as_type_ptr()
    }
    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        object
            .hasattr(intern!(object.py(), "__reversed__"))
            .unwrap_or_else(|err| false_and_write(err, object))
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
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
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
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
                .unwrap_or_else(|err| false_and_write(err, object))
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
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
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
                .unwrap_or_else(|err| false_and_write(err, object))
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
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
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
                .unwrap_or_else(|err| false_and_write(err, object))
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
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, "collections", "deque")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        object
            .is_instance(&Self::type_object(object.py()).into_any())
            .unwrap_or_else(|err| false_and_write(err, object))
    }
}

impl PyDeque {
    #[inline]
    pub fn new<'py>(
        iterable: Bound<'py, PyAny>,
        maxlen: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::type_object(iterable.py())
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
    fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, COLLECTIONS_ABC, "SupportsItems")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        object
            .hasattr(intern!(object.py(), "items"))
            .unwrap_or_else(|err| false_and_write(err, object))
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
        fn type_object_raw(py: Python<'_>) -> *mut ffi::PyTypeObject {
            static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
            TYPE.import(py, "itertools", "repeat")
                .unwrap()
                .as_type_ptr()
        }

        #[inline]
        fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
            object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| false_and_write(err, object))
        }
    }
    impl PyRepeat {
        #[inline(always)]
        pub fn new<'py>(
            obj: &Bound<'py, PyAny>,
            n: Option<&Bound<'py, PyInt>>,
        ) -> PyResult<Bound<'py, Self>> {
            let py = obj.py();
            Self::type_object(py)
                .pipe(|func| match n {
                    Some(n) => func.call1((obj, n)),
                    None => func.call1((obj,)),
                })
                .map(|obj| unsafe { obj.cast_into_unchecked::<Self>() })
        }
    }
}
fn false_and_write(err: PyErr, object: &Bound<'_, PyAny>) -> bool {
    err.write_unraisable(object.py(), Some(object));
    false
}
