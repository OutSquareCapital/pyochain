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
pub(super) trait ListGetter:
    PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    type T: Deref<Target = InnerData> + DerefMut + ListDataOwner + PyRepr;
    type I: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::Bounded<Self::T>>;
    type IRev: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::BoundedRev<Self::T>>;
    type IFull: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::Full<Self::T>>;
    type IFullRev: IntoInit + PyClass<BaseType = abc::PyoIterator> + From<rsiter::FullRev<Self::T>>;
    fn inner(&self) -> &Arc<Mutex<Self::T>>;
    #[inline(always)]
    fn lock(&self) -> MutexGuard<'_, Self::T> {
        self.inner().try_into_inner()
    }
    fn is(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.inner(), other.inner())
    }
    fn iter_bounds<'py>(
        &self,
        py: Python<'py>,
        bounds: Option<Bounds>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        match (bounds, reverse) {
            (None, _) => iterators::Iter::empty(py).map(Bound::into_super),
            (Some(bounds), true) => rsiter::BoundedRev::new(self.inner().clone(), bounds)
                .conv::<Self::IRev>()
                .into_bound(py)
                .map(Bound::into_super),
            (Some(bounds), false) => rsiter::Bounded::new(self.inner().clone(), bounds)
                .conv::<Self::I>()
                .into_bound(py)
                .map(Bound::into_super),
        }
    }
}
impl ListGetter for sorted::SortedList {
    type T = ListsData;
    type I = iter::PyBounded;
    type IRev = iter::PyBoundedRev;
    type IFull = iter::PyFull;
    type IFullRev = iter::PyFullRev;
    #[inline(always)]
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
impl ListGetter for sorted::SortedKeyList {
    type T = KeysListsData;
    type I = iter::PyBoundedKey;
    type IRev = iter::PyBoundedKeyRev;
    type IFull = iter::PyFullKey;
    type IFullRev = iter::PyFullKeyRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
impl ListGetter for sorted::SortedSet {
    type T = SetData<ListsData>;
    type I = iter::PySetBounded;
    type IRev = iter::PySetBoundedRev;
    type IFull = iter::PySetFull;
    type IFullRev = iter::PySetFullRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
impl ListGetter for sorted::SortedKeySet {
    type T = SetData<KeysListsData>;
    type I = iter::PySetBoundedKey;
    type IRev = iter::PySetBoundedKeyRev;
    type IFull = iter::PySetFullKey;
    type IFullRev = iter::PySetFullKeyRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}

impl ListGetter for sorted::SortedDict {
    type T = DictData<ListsData>;
    type I = iter::PyDictBounded;
    type IRev = iter::PyDictBoundedRev;
    type IFull = iter::PyDictFull;
    type IFullRev = iter::PyDictFullRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
impl ListGetter for sorted::SortedKeyDict {
    type T = DictData<KeysListsData>;
    type I = iter::PyDictBoundedKey;
    type IRev = iter::PyDictBoundedKeyRev;
    type IFull = iter::PyDictFullKey;
    type IFullRev = iter::PyDictFullKeyRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
