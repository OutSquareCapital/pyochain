use std::ops::{Deref, DerefMut};

use crate::{DictData, InnerData, SetData, prelude::*};

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

#[macro_export]
macro_rules! impl_inner_getter {
    ($name:ident) => {
        impl std::ops::Deref for $name {
            type Target = InnerData;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl std::ops::DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
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
