use crate::{InnerData, prelude::*, types::VecPy};
use derive_more::{Constructor, Deref, DerefMut};
use pyo3::{
    prelude::*,
    types::{PyDict, PySet},
};
use pyo3_ext::types::PyFrozenClass;
use std::ops::{Deref, DerefMut};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
#[derive(Default, Deref, DerefMut)]
pub struct ListsData(pub(super) InnerData);
#[derive(Deref, DerefMut)]
pub struct KeysListsData(
    #[deref]
    #[deref_mut]
    pub(super) InnerData,
    pub Vec<VecPy>,
    pub(super) Py<PyAny>,
);
pub struct SetData<T: ListsDataMethods>(pub(super) T, pub Py<PySet>);
#[derive(Constructor)]
pub struct DictData<T: ListsDataMethods>(pub(super) T, pub Py<PyDict>);
macro_rules! impl_getters {
    ($name:ident) => {
        impl<T: ListsDataMethods> Deref for $name<T> {
            type Target = InnerData;
            fn deref(&self) -> &InnerData {
                self.0.deref()
            }
        }
        impl<T: ListsDataMethods> DerefMut for $name<T> {
            fn deref_mut(&mut self) -> &mut InnerData {
                self.0.deref_mut()
            }
        }
        impl<T: ListsDataMethods> AsRef<T> for $name<T> {
            fn as_ref(&self) -> &T {
                &self.0
            }
        }
        impl<T: ListsDataMethods> AsMut<T> for $name<T> {
            fn as_mut(&mut self) -> &mut T {
                &mut self.0
            }
        }
    };
}
macro_rules! impl_self_as_ref {
    ($($name:ty),+) => {
        $(
            impl AsRef<$name> for $name {
                #[inline(always)]
                fn as_ref(&self) -> &$name {
                    self
                }
            }
            impl AsMut<$name> for $name {
                #[inline(always)]
                fn as_mut(&mut self) -> &mut $name {
                    self
                }
            }
        )+
    };
}
impl_self_as_ref!(ListsData, KeysListsData);
impl_getters!(SetData);
impl_getters!(DictData);

pub trait ListGetter: PyFrozenClass + AsRef<RwLock<Self::T>> {
    type L: ListsDataMethods;
    type T: Deref<Target = InnerData>
        + DerefMut
        + AsRef<Self::L>
        + AsMut<Self::L>
        + PyRepr
        + Send
        + Sync;
    #[inline(always)]
    fn read(&self) -> RwLockReadGuard<'_, Self::T> {
        self.as_ref().try_read().unwrap()
    }
    #[inline(always)]
    fn write(&self) -> RwLockWriteGuard<'_, Self::T> {
        self.as_ref().try_write().unwrap()
    }
}
