use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, MutexGuard},
};

use crate::{
    abc,
    collections::sorted::{self, iter},
    core::iterators,
    traits::IntoInit,
};
use pyo3::{PyClass, prelude::*};
use sorted_rs::{
    Bounds, DictData, InnerData, KeysListsData, ListsData, SetData, iter as rsiter, prelude::*,
};
use std_tools::prelude::*;
use tap::Conv;
pub trait ListGetter:
    Sync + Send + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + AsRef<Arc<Mutex<Self::T>>>
{
    type L: ListsDataMethods;
    type T: Deref<Target = InnerData> + DerefMut + ListDataOwner<List = Self::L> + PyRepr;
    type I: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::Bounded<Self::T>>;
    type IRev: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::BoundedRev<Self::T>>;
    type IFull: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::Full<Self::T>>;
    type IFullRev: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::FullRev<Self::T>>;
    #[inline(always)]
    fn lock(&self) -> MutexGuard<'_, Self::T> {
        self.as_ref().try_into_inner()
    }
    fn iter_bounds<'py>(
        &self,
        py: Python<'py>,
        bounds: Option<Bounds>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        match (bounds, reverse) {
            (None, _) => iterators::Iter::empty(py).map(Bound::into_super),
            (Some(bounds), true) => rsiter::BoundedRev::new(self.as_ref().clone(), bounds)
                .conv::<Self::IRev>()
                .into_bound(py)
                .map(Bound::into_super),
            (Some(bounds), false) => rsiter::Bounded::new(self.as_ref().clone(), bounds)
                .conv::<Self::I>()
                .into_bound(py)
                .map(Bound::into_super),
        }
    }
}

macro_rules! impl_arc_as_ref {
    ($($t:ty),* $(,)?) => {
        $(
            impl AsRef<Arc<Mutex<<$t as ListGetter>::T>>> for $t {
                fn as_ref(&self) -> &Arc<Mutex<<$t as ListGetter>::T>> {
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
    type IRev = iter::PyBoundedRev;
    type IFull = iter::PyFull;
    type IFullRev = iter::PyFullRev;
}
impl ListGetter for sorted::SortedKeyList {
    type T = KeysListsData;
    type L = KeysListsData;
    type I = iter::PyBoundedKey;
    type IRev = iter::PyBoundedKeyRev;
    type IFull = iter::PyFullKey;
    type IFullRev = iter::PyFullKeyRev;
}
impl ListGetter for sorted::SortedSet {
    type T = SetData<ListsData>;
    type L = ListsData;
    type I = iter::PySetBounded;
    type IRev = iter::PySetBoundedRev;
    type IFull = iter::PySetFull;
    type IFullRev = iter::PySetFullRev;
}
impl ListGetter for sorted::SortedKeySet {
    type T = SetData<KeysListsData>;
    type L = KeysListsData;
    type I = iter::PySetBoundedKey;
    type IRev = iter::PySetBoundedKeyRev;
    type IFull = iter::PySetFullKey;
    type IFullRev = iter::PySetFullKeyRev;
}
impl ListGetter for sorted::SortedDict {
    type T = DictData<ListsData>;
    type L = ListsData;
    type I = iter::PyDictBounded;
    type IRev = iter::PyDictBoundedRev;
    type IFull = iter::PyDictFull;
    type IFullRev = iter::PyDictFullRev;
}
impl ListGetter for sorted::SortedKeyDict {
    type T = DictData<KeysListsData>;
    type L = KeysListsData;
    type I = iter::PyDictBoundedKey;
    type IRev = iter::PyDictBoundedKeyRev;
    type IFull = iter::PyDictFullKey;
    type IFullRev = iter::PyDictFullKeyRev;
}
