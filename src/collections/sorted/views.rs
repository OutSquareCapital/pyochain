use std::sync::{Arc, Mutex, MutexGuard};

use crate::{
    abc,
    collections::sorted::{SortedDict, SortedKeyDict, SortedSet, core::ObjOrVec},
    traits::IntoInit,
};
use pyo3::{PyClass, PyTypeInfo, prelude::*};
use pyo3_ext::prelude::*;
use pyochain_macros::py_abc;
use sorted_rs::{
    DictData, KeysListsData, ListsData, SetData, prelude::*, types::DictDataRef, views,
};
use std_tools::prelude::*;
use tap::prelude::*;
type DictRef<T> = Arc<Mutex<DictData<T>>>;

macro_rules! impl_base_sorted_view {
    ($($l:ty:$name:ty => [$($getitem:path => $t:ident),* $(,)?] );* $(;)?) => {
        $(
            $(
                #[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoSequence, sequence)]
                pub struct $t(DictRef<$l>);

                impl From<Arc<Mutex<DictData<$l>>>> for $t {
                fn from(mapping: Arc<Mutex<DictData<$l>>>) -> Self {
                    Self(mapping)
                    }
                }

                impl SortedViewMethods for $t {
                        type L = $l;
                        type M = $name;
                        fn mapping(&self) -> MutexGuard<'_, DictData<Self::L>> {
                            self.0.try_into_inner()
                        }
                        fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py> {
                            $getitem(&mut self.mapping(), index).and_then_left(|x|x.try_into_py())
                        }
                }
            )*
        )*
    };
}
impl_base_sorted_view!(
    ListsData: SortedDict => [
        views::get_item_for_items => SortedItemsView,
        views::get_item_for_values => SortedValuesView,
        views::get_item_for_keys => SortedKeysView,
    ];
    KeysListsData: SortedKeyDict => [
        views::get_item_for_items => SortedByKeyItemsView,
        views::get_item_for_values => SortedByKeyValuesView,
        views::get_item_for_keys => SortedByKeyKeysView,
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
        SetData::try_from(it)?.conv::<SortedSet>().into_bound(py)
    }
}

#[py_abc(
    SortedItemsView,
    SortedKeysView,
    SortedValuesView,
    SortedByKeyItemsView,
    SortedByKeyKeysView,
    SortedByKeyValuesView
)]
pub trait SortedViewMethods:
    PyClass<BaseType = abc::PyoSequence> + From<DictDataRef<Self::L>>
where
    DictData<Self::L>: PyRepr,
{
    type L: ListsDataMethods;
    type M: PyTypeInfo;
    #[skip]
    fn mapping(&self) -> MutexGuard<'_, DictData<Self::L>>;
    fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py>;
    fn __len__(&self, py: Python<'_>) -> usize {
        self.mapping().__len__(py)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = Self::type_object(py).name()?;
        let values = self.mapping().repr::<Self::M>(py)?;
        Ok(format!("{name}({values})"))
    }
    fn __delitem__(&self, index: Bound<'_, PyAny>) -> PyResult<()> {
        views::delitem(&mut self.mapping(), index)
    }
}
