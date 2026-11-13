use std::ops::Deref;

use crate::{
    abc,
    collections::sorted::{SortedDict, SortedKeyDict, SortedSet, core::ObjOrVec},
};
use derive_more::{Deref, From};

use pyo3::{PyClass, prelude::*};
use pyo3_ext::prelude::*;
use pyochain_macros::py_abc;
use sorted_rs::{prelude::*, types, views};
use std_tools::prelude::*;
use tap::prelude::*;
macro_rules! impl_base_sorted_view {
    ($($inner:ty => $pycls:ident),* $(,)?) => {
        $(
            #[derive(From, Deref)]
            #[from(forward)]
            #[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoSequence, sequence)]
            pub struct $pycls($inner);
        )*
    };
}
impl_base_sorted_view!(
    views::OnItems<SortedDict> => SortedItemsView,
    views::OnValues<SortedDict> => SortedValuesView,
    views::OnKeys<SortedDict> => SortedKeysView,
    views::OnItems<SortedKeyDict> => SortedByKeyItemsView,
    views::OnValues<SortedKeyDict> => SortedByKeyValuesView,
    views::OnKeys<SortedKeyDict> => SortedByKeyKeysView,
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
        types::Set::try_from(it)?.conv::<SortedSet>().into_bound(py)
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
pub trait PySortedViewMethods:
    PyClass<BaseType = abc::PyoSequence> + Deref<Target: ViewMethods>
{
    fn __getitem__<'py>(&self, py: Python<'py>, index: types::IntOrSlice<'py>) -> ObjOrVec<'py> {
        self.get_item(py, index)
            .and_then_left(TryIntoPy::try_into_py)
    }
    fn __len__(&self, py: Python<'_>) -> usize {
        self.len(py)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.deref().repr_from::<Self>(py)
    }
    fn __delitem__(&self, py: Python<'_>, index: types::IntOrSlice<'_>) -> PyResult<()> {
        self.del_item(py, index)
    }
}
