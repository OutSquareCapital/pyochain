use either::Either;
use pyo3::{
    PyClass,
    prelude::*,
    types::{PyList, PySlice, PyTuple, PyType},
};
use pyo3_ext::iter::{CollectBoundIterator, TryCollectBoundIterator};
use pyochain_macros::{py_abc, try_cast_into};

use crate::{
    abc,
    collections::{
        SortedDict, SortedKeyDict, SortedSet,
        sorted::traits::{BaseSortedDict, BaseSortedList, ListGetter, ObjOrVec, SortedListGetters},
    },
    traits::{IntoPyochain, PyoABC},
};
use tap::prelude::*;
#[py_abc(
    SortedItemsView,
    SortedKeysView,
    SortedValuesView,
    SortedByKeyItemsView,
    SortedByKeyKeysView,
    SortedByKeyValuesView
)]
pub trait BaseSortedView:
    Sized + PyClass<BaseType = abc::PyoSequence> + abc::traits::MappingView + Send + Sync
where
    Self::M: BaseSortedDict + PyClass,
{
    #[skip]
    fn new(mapping: Bound<'_, Self::M>) -> Self;
    #[skip]
    fn into_bound(self, py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        abc::PyoSequence::build_init()
            .add_subclass(self)
            .pipe(|initializer| Bound::new(py, initializer))
    }
    fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py>;
    fn __delitem__(&self, index: Bound<'_, PyAny>) -> PyResult<()> {
        let py = index.py();
        let mapping = self.mapping().get();
        let dict = mapping.get_inner().bind(py);
        let list = mapping.get_list().get();
        try_cast_into! {
            match index {
                Case::PySlice(slice) => {
                    let keys = list.get_data().getitem_from_slice(py, &slice)?;
                    list.delitem_from_slice(py, slice)?;
                    for key in keys {
                        dict.del_item(key)?;
                    }
                    Ok(())
                },
                int => {
                    let key = list.pop(py, int.extract::<isize>()?)?;
                    dict.del_item(key)?;
                    Ok(())
                }
            }
        }
    }
}

#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoSequence, sequence)]
pub struct SortedKeysView(Py<SortedDict>);

#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoSequence, sequence)]
pub struct SortedByKeyKeysView(Py<SortedKeyDict>);

#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoSequence, sequence)]
pub struct SortedValuesView(Py<SortedDict>);

#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoSequence, sequence)]
pub struct SortedByKeyValuesView(Py<SortedKeyDict>);

#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoSequence, sequence)]
pub struct SortedItemsView(Py<SortedDict>);

#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoSequence, sequence)]
pub struct SortedByKeyItemsView(Py<SortedKeyDict>);

macro_rules! impl_mapping_view_for_sorted_view {
    ($($t:ty => $i:ty),*) => {
        $(
            impl abc::traits::MappingView for $t {
                type M = $i;
                fn mapping(&self) -> &Py<Self::M> {
                    &self.0
                }
                fn __len__(&self, py: Python<'_>) -> usize {
                    self.mapping().get().len(py)
                }
            }
        )*
    };
}
macro_rules! impl_base_sorted_view_for_items {
    ($($t:ty),*) => {
        $(
            impl BaseSortedView for $t {
                fn new(mapping: Bound<'_, Self::M>) -> Self {
                    Self(mapping.unbind())
                }
                fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py> {
                    get_item_for_items_view(self, index)
                }
            }
        )*
    };
}
macro_rules! impl_base_sorted_view_for_values {
    ($($t:ty),*) => {
        $(
            impl BaseSortedView for $t {
                fn new(mapping: Bound<'_, Self::M>) -> Self {
                    Self(mapping.unbind())
                }
                fn into_bound(self, py: Python<'_>) -> PyResult<Bound<'_, Self>> {
                    abc::PyoSequence::build_init()
                        .add_subclass(self)
                        .pipe(|initializer| Bound::new(py, initializer))
                }
            fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py> {
                get_item_for_values_view(self, index)
            }
            }
        )*
    };
}
macro_rules! impl_base_sorted_view_for_keys {
    ($($t:ty),*) => {
        $(
            impl BaseSortedView for $t {
                fn new(mapping: Bound<'_, Self::M>) -> Self {
                    Self(mapping.unbind())
                }
                fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py> {
                    get_item_for_key_view(self, index)
            }}
        )*
    };
}
impl_mapping_view_for_sorted_view!(
    SortedItemsView => SortedDict,
    SortedKeysView => SortedDict,
    SortedValuesView => SortedDict,
    SortedByKeyItemsView => SortedKeyDict,
    SortedByKeyKeysView => SortedKeyDict,
    SortedByKeyValuesView => SortedKeyDict
);
impl_base_sorted_view_for_items!(SortedItemsView, SortedByKeyItemsView);
impl_base_sorted_view_for_values!(SortedValuesView, SortedByKeyValuesView);
impl_base_sorted_view_for_keys!(SortedKeysView, SortedByKeyKeysView);
macro_rules! impl_from_iterable {
    ($($t:ty),*) => {
        $(
        #[pymethods]
        impl $t {
            #[classmethod]
            fn _from_iterable<'py>(
                cls: Bound<'py, PyType>,
                it: Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, SortedSet>> {
                SortedSet::from_iterable(it)?.into_bound(cls.py())
            }
        }
        )*
    };
}
impl_from_iterable!(
    SortedItemsView,
    SortedKeysView,
    SortedByKeyItemsView,
    SortedByKeyKeysView
);
#[inline(always)]
fn get_item_for_items_view<'py, T: BaseSortedView<M: BaseSortedDict>>(
    slf: &T,
    index: Bound<'py, PyAny>,
) -> ObjOrVec<'py> {
    let py = index.py();
    let mapping = slf.mapping().get();
    let dict = mapping.get_inner().bind(index.py()).as_any();
    let mut mapping_list = mapping.get_list().get().get_data();

    try_cast_into! {
        match index {
            Case::PySlice(slice) => mapping_list
                .getitem_from_slice(py, &slice)?
                .iter()
                .map(|key| {
                    PyTuple::new(py, [key.bind(py), &dict.get_item(key)?]).map(Bound::into_any)
                })
                .try_collect_bound::<PyList>(py)?
                .into_pyochain()
                .map(Either::Right),
            int => {
                let key = mapping_list.getitem_from_int(py, int.extract::<isize>()?)?;
                let value = dict.get_item(&key)?;
                PyTuple::new(py, [key, value])
                    .map(Bound::into_any)
                    .map(Either::Left)
            }
        }
    }
}
#[inline(always)]
fn get_item_for_values_view<'py, T: BaseSortedView<M: BaseSortedDict>>(
    slf: &T,
    index: Bound<'py, PyAny>,
) -> ObjOrVec<'py> {
    let py = index.py();
    let mapping = slf.mapping().get();
    let dict = mapping.get_inner().bind(py).as_any();
    let mut mapping_list = mapping.get_list().get().get_data();

    try_cast_into! {
        match index {
            Case::PySlice(slice) => mapping_list
                .getitem_from_slice(py, &slice)?
                .iter()
                .map(|key| dict.get_item(key))
                .try_collect_bound::<PyList>(py)?
                .into_pyochain()
                .map(Either::Right),
            int => dict
                .get_item(mapping_list.getitem_from_int(py, int.extract::<isize>()?)?)
                .map(Either::Left),
        }
    }
}
#[inline(always)]
fn get_item_for_key_view<'py, T: BaseSortedView<M: BaseSortedDict>>(
    slf: &T,
    index: Bound<'py, PyAny>,
) -> ObjOrVec<'py> {
    let py = index.py();
    let mut mapping_list = slf.mapping().get().get_list().get().get_data();

    try_cast_into! {
        match index {
            Case::PySlice(slice) => mapping_list
                .getitem_from_slice(py, &slice)?
                .iter()
                .collect_bound::<PyList>(py)?
                .into_pyochain()
                .map(Either::Right),
            int => mapping_list
                .getitem_from_int(py, int.extract::<isize>()?)
                .map(Either::Left),
        }
    }
}
