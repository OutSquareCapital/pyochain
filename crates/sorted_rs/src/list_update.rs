use derive_more::From;
use itertools::Itertools;
use parking_lot::RwLock;
use pyo3::{prelude::*, types::PyIterator};
use pyo3_ext::types::PyFrozenClass;
use std_tools::prelude::TryIterator;
use tap::prelude::*;

use crate::{
    inner::InnerList,
    sort::{ExtendKind, SortedByKeys, SortedVec},
    traits::ListsMethods,
    types::VecPy,
};

pub trait PyListsDataRef: PyFrozenClass + AsRef<RwLock<Self::List>> + From<Self::List> {
    type List: ListsMethods;
}
#[derive(From)]
pub enum IntoList {
    Vec(VecPy),
    PyIter(Py<PyIterator>),
}
impl IntoList {
    #[inline]
    pub fn try_new<R: PyListsDataRef>(right: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = right.py();
        match right.cast_exact::<R>().ok().map(Bound::get) {
            Some(list) => list
                .as_ref()
                .read()
                .values
                .iter_owned(py)
                .collect_vec()
                .pipe(Self::Vec)
                .pipe(Ok),
            None => right.try_iter().map(Bound::unbind).map(Self::PyIter),
        }
    }
    pub fn into_new_list<T: ListsMethods>(self, py: Python<'_>, list: &T) -> PyResult<T> {
        let mut new_inst = list.new_empty(py);
        self.extend_into(py, &mut new_inst)?;
        Ok(new_inst)
    }
    pub fn concat_into<T: ListsMethods>(self, py: Python<'_>, list: &T) -> PyResult<T> {
        let mut new_inst = list.new_empty(py);
        let key = list.key_func(py);
        let slf = list.values.iter_owned(py);
        match self {
            Self::PyIter(it) => slf
                .map(|x| Ok(x.into_bound(py)))
                .chain(it.into_bound(py))
                .sorted_by_pykey(&key),
            Self::Vec(vec) => slf.chain(vec).sorted_by_pykey(&key),
        }
        .map(ExtendKind::Big)?
        .extend_into(py, &mut new_inst)?;
        Ok(new_inst)
    }
    #[inline]
    pub fn extend_into(self, py: Python<'_>, list: &mut impl ListsMethods) -> PyResult<()> {
        //TODO: Check `itertools::merge` to see if it can be beneficial here.
        match self {
            _ if list.is_empty() => Strategy::Empty(self),
            Self::Vec(vec) => Strategy::from_vec(vec, list),
            Self::PyIter(it) => it
                .into_bound(py)
                .map_ok(Bound::unbind)
                .collect_with_capacity::<PyResult<VecPy>>()
                .map(|vec| Strategy::from_vec(vec, list))?,
        }
        .sorted(py, list)?
        .extend_into(py, list)
    }
}
enum Strategy {
    Empty(IntoList),
    Size(ExtendKind<VecPy>),
}
impl Strategy {
    /// Treshold on the number of values to add **as a whole**.
    const MIN_ITEMS: usize = 4;
    fn from_vec(vec: VecPy, list: &InnerList) -> Self {
        let length = vec.len();
        if 4 * length >= list.len() {
            Self::Size(ExtendKind::Big(vec))
        } else if length >= Self::MIN_ITEMS {
            Self::Size(ExtendKind::Medium(vec))
        } else {
            Self::Size(ExtendKind::Small(vec))
        }
    }
    fn sorted<L: ListsMethods>(
        self,
        py: Python<'_>,
        list: &L,
    ) -> PyResult<ExtendKind<SortedVec<L::Item>>> {
        let key = list.key_func(py);
        match self {
            Self::Empty(IntoList::Vec(vec)) => vec.sorted_by_pykey(&key).map(ExtendKind::Big),
            Self::Empty(IntoList::PyIter(it)) => {
                it.into_bound(py).sorted_by_pykey(&key).map(ExtendKind::Big)
            }
            Self::Size(ExtendKind::Big(vec)) => list
                .values
                .iter_owned(py)
                .chain(vec)
                .sorted_by_pykey(&key)
                .map(ExtendKind::Big),
            Self::Size(ExtendKind::Medium(vec)) => {
                vec.sorted_by_pykey(&key).map(ExtendKind::Medium)
            }
            Self::Size(ExtendKind::Small(vec)) => vec.sorted_by_pykey(&key).map(ExtendKind::Small),
        }
    }
}
