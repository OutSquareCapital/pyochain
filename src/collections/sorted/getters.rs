use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use crate::{
    abc,
    collections::sorted::{self, core::IterRes, iter},
    traits::IntoInit,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use pyo3::{PyClass, prelude::*};
use sorted_rs::{
    Bounds, DictData, InnerData, KeysListsData, ListsData, SetData, iter as rsiter, prelude::*,
};
use tap::prelude::*;
pub trait ListGetter:
    Sync + Send + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + AsRef<Arc<RwLock<Self::T>>>
{
    type L: ListsDataMethods;
    type T: Deref<Target = InnerData> + DerefMut + ListDataOwner<List = Self::L> + PyRepr;
    type I: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::Bounded<Self::T>>;
    #[inline(always)]
    fn lock(&self) -> RwLockReadGuard<'_, Self::T> {
        self.as_ref().read()
    }
    #[inline(always)]
    fn write(&self) -> RwLockWriteGuard<'_, Self::T> {
        self.as_ref().write()
    }
    fn iter_bounds<'py>(
        &self,
        py: Python<'py>,
        bounds: Option<Bounds>,
        reverse: bool,
    ) -> IterRes<'py> {
        self.as_ref()
            .clone()
            .pipe(|x| rsiter::Bounded::new(x, bounds.unwrap_or_default().into(), reverse))
            .conv::<Self::I>()
            .into_bound(py)
            .map(Bound::into_super)
    }
}

macro_rules! impl_arc_as_ref {
    ($($t:ty),* $(,)?) => {
        $(
            impl AsRef<Arc<RwLock<<$t as ListGetter>::T>>> for $t {
                fn as_ref(&self) -> &Arc<RwLock<<$t as ListGetter>::T>> {
                    &self.0
                }
            }
        )*
    };
}
impl_arc_as_ref!(
    sorted::SortedList,
    sorted::SortedKeyList,
    sorted::SortedSet,
    sorted::SortedKeySet,
    sorted::SortedDict,
    sorted::SortedKeyDict
);
impl ListGetter for sorted::SortedList {
    type T = ListsData;
    type L = ListsData;
    type I = iter::PyBounded;
}
impl ListGetter for sorted::SortedKeyList {
    type T = KeysListsData;
    type L = KeysListsData;
    type I = iter::PyBoundedKey;
}
impl ListGetter for sorted::SortedSet {
    type T = SetData<ListsData>;
    type L = ListsData;
    type I = iter::PySetBounded;
}
impl ListGetter for sorted::SortedKeySet {
    type T = SetData<KeysListsData>;
    type L = KeysListsData;
    type I = iter::PySetBoundedKey;
}
impl ListGetter for sorted::SortedDict {
    type T = DictData<ListsData>;
    type L = ListsData;
    type I = iter::PyDictBounded;
}
impl ListGetter for sorted::SortedKeyDict {
    type T = DictData<KeysListsData>;
    type L = KeysListsData;
    type I = iter::PyDictBoundedKey;
}
