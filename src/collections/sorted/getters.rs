use std::ops::{Deref, DerefMut};

use crate::{
    collections::sorted::{self, core::IterRes, iter},
    traits::IntoInit,
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use pyo3::prelude::*;
use pyo3_ext::prelude::*;
use sorted_rs::{
    DictData, InnerData, KeysListsData, ListsData, SetData,
    iter::{Iter, IterBounded, IterBoundedRev, IterKind, IterRev},
    prelude::*,
};
use tap::prelude::*;
pub trait ListGetter: PyFrozenClass + AsRef<RwLock<Self::T>> {
    type L: ListsDataMethods;
    type T: Deref<Target = InnerData>
        + DerefMut
        + ListDataOwner<List = Self::L>
        + PyRepr
        + Send
        + Sync;
    #[inline(always)]
    fn lock(&self) -> RwLockReadGuard<'_, Self::T> {
        self.as_ref().read()
    }
    #[inline(always)]
    fn write(&self) -> RwLockWriteGuard<'_, Self::T> {
        self.as_ref().write()
    }
    #[inline(always)]
    fn build_iter<'py>(slf: &Bound<'py, Self>, kind: IterKind) -> IterRes<'py> {
        let py = slf.py();
        (slf.get().lock(), slf.clone().into_any().unbind()).pipe(|x| match kind {
            IterKind::Fwd => x
                .conv::<Iter>()
                .conv::<iter::SortedIter>()
                .into_bound(py)
                .map(Bound::into_super),
            IterKind::Rev => x
                .conv::<IterRev>()
                .conv::<iter::SortedIterRev>()
                .into_bound(py)
                .map(Bound::into_super),
            IterKind::Bounded(bounds) => IterBounded::new(x, bounds)
                .conv::<iter::SortedIterBounded>()
                .into_bound(py)
                .map(Bound::into_super),
            IterKind::BoundedRev(bounds) => IterBoundedRev::new(x, bounds)
                .conv::<iter::SortedIterBoundedRev>()
                .into_bound(py)
                .map(Bound::into_super),
        })
    }
}
macro_rules! impl_arc_as_ref {
    ($($t:ty),* $(,)?) => {
        $(
            impl AsRef<RwLock<<$t as ListGetter>::T>> for $t {
                fn as_ref(&self) -> &RwLock<<$t as ListGetter>::T> {
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
}
impl ListGetter for sorted::SortedKeyList {
    type T = KeysListsData;
    type L = KeysListsData;
}
impl ListGetter for sorted::SortedSet {
    type T = SetData<ListsData>;
    type L = ListsData;
}
impl ListGetter for sorted::SortedKeySet {
    type T = SetData<KeysListsData>;
    type L = KeysListsData;
}
impl ListGetter for sorted::SortedDict {
    type T = DictData<ListsData>;
    type L = ListsData;
}
impl ListGetter for sorted::SortedKeyDict {
    type T = DictData<KeysListsData>;
    type L = KeysListsData;
}
