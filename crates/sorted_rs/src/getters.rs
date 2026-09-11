use crate::{ListsDataMethods, inner::InnerData, types::VecPy};
use pyo3::prelude::*;

pub trait InnerGetter: Sized {
    fn inner(&self) -> &InnerData;
    fn inner_mut(&mut self) -> &mut InnerData;
}

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

pub trait ListDataGetters: Sized {
    fn lists(&self) -> &[VecPy];
    fn lists_mut(&mut self) -> &mut Vec<VecPy>;
    fn maxes(&self) -> &[Py<PyAny>];
    fn maxes_mut(&mut self) -> &mut VecPy;
    fn idx(&self) -> &[usize];
    fn idx_mut(&mut self) -> &mut Vec<usize>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn increment_len(&mut self);
    fn decrement_len(&mut self);
    fn set_len(&mut self, len: usize);
    fn offset(&self) -> usize;
    fn set_offset(&mut self, offset: usize);
    fn load(&self) -> usize;
    fn set_load(&mut self, load: usize);
}

impl<T: InnerGetter> ListDataGetters for T {
    fn lists(&self) -> &[VecPy] {
        &self.inner().lists
    }
    fn lists_mut(&mut self) -> &mut Vec<VecPy> {
        &mut self.inner_mut().lists
    }
    fn maxes(&self) -> &[Py<PyAny>] {
        &self.inner().maxes
    }
    fn maxes_mut(&mut self) -> &mut VecPy {
        &mut self.inner_mut().maxes
    }
    fn idx(&self) -> &[usize] {
        &self.inner().idx
    }
    fn idx_mut(&mut self) -> &mut Vec<usize> {
        &mut self.inner_mut().idx
    }
    fn len(&self) -> usize {
        self.inner().len
    }
    fn set_len(&mut self, len: usize) {
        self.inner_mut().len = len;
    }
    fn increment_len(&mut self) {
        self.inner_mut().len += 1;
    }
    fn decrement_len(&mut self) {
        self.inner_mut().len -= 1;
    }
    fn offset(&self) -> usize {
        self.inner().offset
    }
    fn set_offset(&mut self, offset: usize) {
        self.inner_mut().offset = offset;
    }
    fn load(&self) -> usize {
        self.inner().load
    }
    fn set_load(&mut self, load: usize) {
        self.inner_mut().load = load;
    }
}

#[macro_export]
macro_rules! impl_inner_getter {
    ($name:ident) => {
        impl $crate::getters::InnerGetter for $name {
            fn inner(&self) -> &InnerData {
                &self.0
            }

            fn inner_mut(&mut self) -> &mut InnerData {
                &mut self.0
            }
        }
    };
}
