///! Traits extending the functionality of various pre-existing Pyo3 types
use pyo3::{
    PyTypeInfo,
    call::PyCallArgs,
    ffi, intern,
    prelude::*,
    types::{
        PyBool, PyDict, PyDictItems, PyDictKeys, PyDictValues, PyFrozenSet, PyInt, PyIterator,
        PyList, PyMapping, PyRange, PySequence, PySet, PyTuple,
    },
};

use crate::types::{
    PopResult, PyAbstractSet, PyDeque, PyIterable, PyMappingView, PyMutableSequence, PyMutableSet,
};
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
impl ABCRegister<'_> for PyMappingView {}

/// Trait for types that we know can safely be converted into a `PyIterator` (i.e. they implement the `__iter__` method in Python).
pub trait IntoPyIterator<'py> {
    fn iter_py(&self) -> Bound<'py, PyIterator>;
}
impl<'py> IntoPyIterator<'py> for Bound<'py, PyIterator> {
    fn iter_py(&self) -> Bound<'py, PyIterator> {
        self.to_owned()
    }
}
macro_rules! impl_into_py_iterator_for_iterable {
    ($($t:ty),* $(,)?) => {
        $(
            impl<'py> IntoPyIterator<'py> for Bound<'py, $t> {
                /// Returns a `PyIterator` with `unwrap_unchecked`, as we know that the type implements `__iter__` and thus can be safely converted into a `PyIterator`.
                fn iter_py(&self) -> Bound<'py, PyIterator> {
                    unsafe { self.try_iter().unwrap_unchecked() }
                }
            }
        )*
    };
}
impl_into_py_iterator_for_iterable!(
    PyTuple,
    PyList,
    PySet,
    PyDict,
    PyFrozenSet,
    PyRange,
    PyDeque,
    PyDictKeys,
    PyDictValues,
    PyDictItems,
    PySequence,
);
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
    fn remove(&self, element: &Bound<'_, PyAny>) -> PyResult<()>;
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
    fn remove(&self, element: &Bound<'_, PyAny>) -> PyResult<()> {
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
#[allow(unused)]
pub trait PyListExtMethods<'py> {
    fn clear(&self) -> ();
    fn extend(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()>;
    fn last(&self) -> PyResult<Bound<'py, PyAny>>;
    fn pop(&self, index: usize) -> PyResult<Bound<'py, PyAny>>;
    fn sort_by(&self, key: &Bound<'_, PyAny>, reverse: bool) -> PyResult<()>;
}
impl<'py> PyListExtMethods<'py> for Bound<'py, PyList> {
    fn pop(&self, index: usize) -> PyResult<Bound<'py, PyAny>> {
        let v = self.get_item(index)?;
        self.del_item(index)?;
        Ok(v)
    }
    fn clear(&self) -> () {
        unsafe { ffi::PyList_Clear(self.as_ptr()) };
    }

    fn extend(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        match unsafe { ffi::PyList_Extend(self.as_ptr(), iterable.as_ptr()) } {
            0 => Ok(()),
            _ => Err(PyErr::fetch(self.py())),
        }
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
pub trait PyDictExtConstructors: PyTypeInfo {
    fn from_keys<'py, T: PyTypeInfo>(
        keys: Bound<'py, T>,
        value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>>;
    fn from_mapping(mapping: Bound<'_, PyMapping>) -> PyResult<Bound<'_, Self>>;
}
impl PyDictExtConstructors for PyDict {
    fn from_keys<'py, T: PyTypeInfo>(
        keys: Bound<'py, T>,
        value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = keys.py();
        Self::type_object(py)
            .call_method1(intern!(py, "fromkeys"), (keys, value))
            .map(|x| unsafe { x.cast_into_unchecked::<PyDict>() })
    }
    fn from_mapping(mapping: Bound<'_, PyMapping>) -> PyResult<Bound<'_, Self>> {
        let dict = PyDict::new(mapping.py());
        dict.update(&mapping).map(|_| dict)
    }
}
#[allow(unused)]
pub trait PyDictExtMethods<'py>: Sized {
    /// Return a view of the dictionnary items, just like calling `dict.items()` in Python
    fn items_view(&self) -> Bound<'py, PyDictItems>;
    /// Return a view of the dictionnary keys, just like calling `dict.keys()` in Python
    fn keys_view(&self) -> Bound<'py, PyDictKeys>;
    /// Return a view of the dictionnary values, just like calling `dict.values()` in Python
    fn values_view(&self) -> Bound<'py, PyDictValues>;
    fn pop(&self, key: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>>;
    fn pop_or_err(&self, key: &Bound<'py, PyAny>) -> PopResult<'py>;
    fn update_from_sequence(&self, seq: &Bound<'py, PyAny>) -> PyResult<&Self>;
    fn merge(self, other: &Bound<'_, PyAny>) -> PyResult<Self>;
}
impl<'py> PyDictExtMethods<'py> for Bound<'py, PyDict> {
    // BUG, TODO: add SupportsKey or something like that Protocol type, bc iterables dom't work with this
    fn merge(self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        match unsafe { ffi::PyDict_Merge(self.as_ptr(), other.as_ptr(), 0) } {
            0 => Ok(self),
            _ => Err(PyErr::fetch(self.py())),
        }
    }
    fn items_view(&self) -> Bound<'py, PyDictItems> {
        unsafe {
            self.call_method0(intern!(self.py(), "items"))
                .unwrap_unchecked()
                .cast_into_unchecked::<PyDictItems>()
        }
    }
    fn keys_view(&self) -> Bound<'py, PyDictKeys> {
        unsafe {
            self.call_method0(intern!(self.py(), "keys"))
                .unwrap_unchecked()
                .cast_into_unchecked::<PyDictKeys>()
        }
    }
    fn values_view(&self) -> Bound<'py, PyDictValues> {
        unsafe {
            self.call_method0(intern!(self.py(), "values"))
                .unwrap_unchecked()
                .cast_into_unchecked::<PyDictValues>()
        }
    }
    /// Remove *key* from the dictionary, and optionally return the removed value.\
    /// Do not raise `KeyError` if the *key* is missing (unlike calling `.call_method1("pop", key)`).\
    /// If the *key* is present, set *result to a new reference to the removed value if result is not NULL, and return `Some(value)`.\
    /// If the *key* is missing, return `None`.
    fn pop(&self, key: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let mut result = core::ptr::null_mut();
        match unsafe { ffi::PyDict_Pop(self.as_ptr(), key.as_ptr(), &mut result) } {
            1 => Ok(Some(unsafe { Bound::from_owned_ptr(self.py(), result) })),
            0 => Ok(None),
            // Return code is -1 here, hence error
            _ => Err(PyErr::fetch(self.py())),
        }
    }
    /// Remove *key* from the dictionary, and return the removed value.\
    /// Raise `KeyError` if the *key* is missing, just like python's `dict.pop(key)` method.\
    fn pop_or_err(&self, key: &Bound<'py, PyAny>) -> PopResult<'py> {
        let mut result = core::ptr::null_mut();
        match unsafe { ffi::PyDict_Pop(self.as_ptr(), key.as_ptr(), &mut result) } {
            1 => PopResult::Ok(unsafe { Bound::from_owned_ptr(self.py(), result) }),
            0 => PopResult::KeyMissing,
            // Return code is -1 here, hence error
            _ => PopResult::Err(PyErr::fetch(self.py())),
        }
    }
    fn update_from_sequence(&self, seq: &Bound<'py, PyAny>) -> PyResult<&Self> {
        match unsafe { ffi::PyDict_MergeFromSeq2(self.as_ptr(), seq.as_ptr(), 1) } {
            0 => Ok(self),
            // Return code is -1 here, hence error
            _ => Err(PyErr::fetch(seq.py())),
        }
    }
}
