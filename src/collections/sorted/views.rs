use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
    abc::PyoSequence,
    collections::{
        SortedSet,
        sorted::traits::{ObjOrVec, SortedViewMethods},
    },
    traits::IntoInit,
};
use either::Either;
use pyo3::{
    prelude::*,
    types::{PyList, PySlice},
};
use pyo3_ext::prelude::*;
use pyochain_macros::{py_abc, try_cast, try_cast_into};
use sorted_rs::{DictData, KeysListsData, ListDataOwner, ListsData, ListsDataMethods};
use std_tools::prelude::*;
type DictRef<T> = Arc<Mutex<DictData<T>>>;

macro_rules! impl_base_sorted_view {
    ($($m:ty:$name:ty => [$($getitem:ident => $t:ident),* $(,)?] );* $(;)?) => {
        $(
            $(
                #[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = PyoSequence, sequence)]
                pub struct $t(DictRef<$m>);

                impl From<Arc<Mutex<DictData<$m>>>> for $t {
                fn from(mapping: Arc<Mutex<DictData<$m>>>) -> Self {
                    Self(mapping)
                    }
                }

                impl SortedViewMethods for $t {
                        type M = $m;
                        const REF_NAME: &'static str = stringify!($name);
                        fn mapping(&self) -> MutexGuard<'_, DictData<Self::M>> {
                            self.0.try_into_inner()
                        }

                        fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py> {
                            $getitem(&mut self.mapping(), index)
                        }
                }
            )*
        )*
    };
}
impl_base_sorted_view!(
    ListsData: SortedDict => [
        get_item_for_items => SortedItemsView,
        get_item_for_values => SortedValuesView,
        get_item_for_keys => SortedKeysView,
    ];
    KeysListsData: SortedKeyDict => [
        get_item_for_items => SortedByKeyItemsView,
        get_item_for_values => SortedByKeyValuesView,
        get_item_for_keys => SortedByKeyKeysView,
    ];
);
#[py_abc(
    SortedItemsView,
    SortedKeysView,
    SortedByKeyItemsView,
    SortedByKeyKeysView
)]
trait FromIterable {
    #[staticmethod]
    #[pyo3(name = "_from_iterable")]
    fn from_iterable(it: Bound<'_, PyAny>) -> PyResult<Bound<'_, SortedSet>> {
        let py = it.py();
        SortedSet::try_from(it)?.into_bound(py)
    }
}

#[inline(always)]
fn get_item_for_items<'py, T: ListsDataMethods>(
    mapping: &mut DictData<T>,
    index: Bound<'py, PyAny>,
) -> ObjOrVec<'py> {
    let py = index.py();
    let dict = mapping.get_dict().clone_ref(py).into_bound(py).into_any();
    try_cast_into! {
        match index {
            Case::PySlice(slice) => mapping
                .list_mut()
                .inner_mut()
                .get_slice(&slice)?
                .iter()
                .map(|key| tuple!(key.bind(py), &dict.get_item(key)?).map(Bound::into_any))
                .try_collect_bound::<PyList>(py)?
                .try_into_py()
                .map(Either::Left),
            int => {
                let key = mapping.extract_index(&int)?;
                let value = dict.get_item(&key)?;
                tuple!(key, value).map(Bound::into_any).map(Either::Right)
            }
        }
    }
}
#[inline(always)]
fn get_item_for_values<'py, T: ListsDataMethods>(
    mapping: &mut DictData<T>,
    index: Bound<'py, PyAny>,
) -> ObjOrVec<'py> {
    let py = index.py();
    let dict = mapping.get_dict().clone_ref(py).into_bound(py).into_any();
    try_cast_into! {
        match index {
            Case::PySlice(slice) => mapping
                .list_mut()
                .inner_mut()
                .get_slice(&slice)?
                .iter()
                .map(|key| dict.get_item(key))
                .try_collect_bound::<PyList>(py)?
                .try_into_py()
                .map(Either::Left),
            int => dict
                .get_item(mapping.extract_index(&int)?)
                .map(Either::Right),
        }
    }
}
#[inline(always)]
fn get_item_for_keys<'py, T: ListsDataMethods>(
    mapping: &mut DictData<T>,
    index: Bound<'py, PyAny>,
) -> ObjOrVec<'py> {
    let py = index.py();
    try_cast! {
        match index {
            Case::PySlice(slice) => mapping
                .list_mut()
                .inner_mut()
                .get_slice(slice)?
                .iter()
                .collect_bound::<PyList>(py)?
                .try_into_py()
                .map(Either::Left),
            int => mapping.extract_index(&int).map(Either::Right),
        }
    }
}
