use pyo3::prelude::*;
/// A `Vec` which contains python objects.
pub type VecPy = Vec<Py<PyAny>>;
pub struct InnerData {
    pub lists: Vec<VecPy>,
    pub maxes: VecPy,
    pub idx: Vec<usize>,
    pub len: usize,
    pub offset: usize,
    pub load: usize,
}
impl Default for InnerData {
    fn default() -> Self {
        Self {
            lists: Vec::default(),
            maxes: Vec::default(),
            idx: Vec::default(),
            len: usize::default(),
            offset: usize::default(),
            load: 1000,
        }
    }
}
pub(super) trait InnerGetter: Sized {
    fn inner(&self) -> &InnerData;
    fn inner_mut(&mut self) -> &mut InnerData;
}

pub trait ListDataGetters: Sized {
    fn lists(&self) -> &Vec<VecPy>;
    fn lists_mut(&mut self) -> &mut Vec<VecPy>;
    fn maxes(&self) -> &VecPy;
    fn maxes_mut(&mut self) -> &mut VecPy;
    fn idx(&self) -> &Vec<usize>;
    fn idx_mut(&mut self) -> &mut Vec<usize>;
    fn length(&self) -> usize;
    fn increment_len(&mut self);
    fn decrement_len(&mut self);
    fn set_len(&mut self, len: usize);
    fn offset(&self) -> usize;
    fn set_offset(&mut self, offset: usize);
    fn load(&self) -> usize;
    fn set_load(&mut self, load: usize);
}

impl<T: InnerGetter> ListDataGetters for T {
    fn lists(&self) -> &Vec<VecPy> {
        &self.inner().lists
    }
    fn lists_mut(&mut self) -> &mut Vec<VecPy> {
        &mut self.inner_mut().lists
    }
    fn maxes(&self) -> &VecPy {
        &self.inner().maxes
    }
    fn maxes_mut(&mut self) -> &mut VecPy {
        &mut self.inner_mut().maxes
    }
    fn idx(&self) -> &Vec<usize> {
        &self.inner().idx
    }
    fn idx_mut(&mut self) -> &mut Vec<usize> {
        &mut self.inner_mut().idx
    }
    fn length(&self) -> usize {
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
        impl $crate::inner::InnerGetter for $name {
            fn inner(&self) -> &InnerData {
                &self.0
            }

            fn inner_mut(&mut self) -> &mut InnerData {
                &mut self.0
            }
        }
    };
}
