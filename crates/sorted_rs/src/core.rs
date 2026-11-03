use crate::{InnerData, prelude::*, types::VecPy};
use derive_more::{AsMut, AsRef, Constructor, Deref, DerefMut};
use pyo3::prelude::*;
use pyo3_ext::types::PyFrozenClass;
use std::ops::{Deref, DerefMut};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
#[derive(Default, Deref, DerefMut)]
pub struct List(pub(super) InnerData);
#[derive(Deref, DerefMut)]
pub struct KeyList(
    #[deref]
    #[deref_mut]
    pub(super) InnerData,
    pub Vec<VecPy>,
    pub(super) Py<PyAny>,
);
#[derive(Constructor, AsRef, AsMut, Deref, DerefMut)]
pub struct Collection<T: ListsMethods, U>(
    #[deref(forward)]
    #[deref_mut(forward)]
    pub(super) T,
    pub Py<U>,
);
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
impl_self_as_ref!(List, KeyList);

pub trait ListGetter: PyFrozenClass + AsRef<RwLock<Self::T>> {
    type L: ListsMethods;
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
