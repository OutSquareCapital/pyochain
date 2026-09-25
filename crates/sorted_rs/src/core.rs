use crate::{InnerData, prelude::*, types::VecPy};
use derive_more::{Deref, DerefMut};
use pyo3::{
    prelude::*,
    types::{PyDict, PySet},
};
use std::ops::{Deref, DerefMut};

#[derive(Default, Deref, DerefMut)]
pub struct ListsData(pub(super) InnerData);
pub struct KeysListsData(pub(super) InnerData, pub Vec<VecPy>, pub(super) Py<PyAny>);
pub struct SetData<T: ListsDataMethods>(pub(super) T, pub Py<PySet>);
pub struct DictData<T: ListsDataMethods>(pub(super) T, pub Py<PyDict>);
pub trait ListDataOwner {
    type List: ListsDataMethods;
    fn list(&self) -> &Self::List;
    fn list_mut(&mut self) -> &mut Self::List;
}
impl<T: ListsDataMethods> ListDataOwner for T {
    type List = T;
    #[inline(always)]
    fn list(&self) -> &Self::List {
        self
    }
    #[inline(always)]
    fn list_mut(&mut self) -> &mut Self::List {
        self
    }
}
impl std::ops::Deref for KeysListsData {
    type Target = InnerData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for KeysListsData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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
        impl<T: ListsDataMethods> ListDataOwner for $name<T> {
            type List = T;
            fn list(&self) -> &Self::List {
                &self.0
            }
            fn list_mut(&mut self) -> &mut Self::List {
                &mut self.0
            }
        }
    };
}

impl_getters!(SetData);
impl_getters!(DictData);
