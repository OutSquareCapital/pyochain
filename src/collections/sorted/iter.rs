use crate::abc;
use derive_more::From;
use pyo3::prelude::*;
use sorted_rs::{DictData, KeysListsData, ListsData, SetData, iter::Iter};
macro_rules! impl_sorted_iter {
    ($($t:ty => $name:ident),+ $(,)?) => {
        $(
            #[derive(From)]
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
            pub struct $name(Iter<$t>);
            impl $name {
                #[inline(always)]
                unsafe extern "C" fn tp_iternext(
                    obj: *mut pyo3::ffi::PyObject,
                ) -> *mut pyo3::ffi::PyObject {
                    let py = unsafe { Python::assume_attached() };
                    unsafe { pyo3::Borrowed::from_ptr(py, obj).cast_unchecked::<Self>().get().0.next() }
                }
                pub fn install_iternext(py: Python<'_>) {
                    unsafe {
                        let ty = <Self as pyo3::type_object::PyTypeInfo>::type_object_raw(py);
                        (*ty).tp_iternext = Some(Self::tp_iternext);
                        pyo3::ffi::PyType_Modified(ty);
                    }
                }
            }
        )+
    };
}

impl_sorted_iter! {
    ListsData => PyBounded,
    KeysListsData => PyBoundedKey,
    SetData<ListsData> => PySetBounded,
    SetData<KeysListsData> => PySetBoundedKey,
    DictData<ListsData> => PyDictBounded,
    DictData<KeysListsData> => PyDictBoundedKey,
}
