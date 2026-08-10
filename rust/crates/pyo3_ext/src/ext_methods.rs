///! Traits extending the functionality of various pre-existing Pyo3 types
use pyo3::{
    PyTypeInfo,
    call::PyCallArgs,
    ffi, intern,
    prelude::*,
    types::{
        PyBool, PyDict, PyDictItems, PyDictKeys, PyDictValues, PyFrozenSet, PyInt, PyIterator,
        PyList, PyRange, PySet, PyTuple,
    },
};
use tap::prelude::*;

use crate::types::{PyAbstractSet, PyDeque, PyIterable, PyMutableSequence, PyMutableSet};
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
