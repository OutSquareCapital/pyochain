use derive_more::From;
use itertools::Itertools;
use parking_lot::RwLock;
use pyo3::{prelude::*, types::PyIterator};
use pyo3_ext::types::PyFrozenClass;
use tap::prelude::*;

use crate::{
    inner::InnerList,
    sort::{ExtendList, SortedByKeys},
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
                .sorted_by_pykey(&key)?,
            Self::Vec(vec) => slf.chain(vec).sorted_by_pykey(&key)?,
        }
        .extend_into(py, &mut new_inst);
        Ok(new_inst)
    }
    #[inline]
    pub fn extend_into(self, py: Python<'_>, list: &mut impl ListsMethods) -> PyResult<()> {
        //TODO: Check `itertools::merge` to see if it can be beneficial here.
        use Strategy as Is;
        let key = list.key_func(py);
        match self.strategy(py, list) {
            Is::Empty(Self::Vec(vec)) => {
                vec.into_iter().sorted_by_pykey(&key)?.extend_into(py, list);
            }
            Is::Empty(Self::PyIter(it)) => it
                .into_bound(py)
                .sorted_by_pykey(&key)?
                .extend_into(py, list),
            Is::Big(Self::Vec(vec)) => list
                .values
                .iter_owned(py)
                .chain(vec)
                .sorted_by_pykey(&key)?
                .extend_into(py, list),
            Is::Big(Self::PyIter(it)) => list
                .values
                .iter_owned(py)
                .map(|x| Ok(x.into_bound(py)))
                .chain(it.into_bound(py))
                .sorted_by_pykey(&key)?
                .extend_into(py, list),
            Is::Small(Self::Vec(vec)) => vec
                .into_iter()
                .map(|x| x.into_bound(py))
                .try_for_each(|val| list.append(val))?,
            Is::Small(Self::PyIter(it)) => {
                it.into_bound(py).try_for_each(|val| list.append(val?))?;
            }
        }
        Ok(())
    }
    fn strategy(self, py: Python<'_>, list: &InnerList) -> Strategy {
        let is_big = |length| 4 * length >= list.len();
        match self {
            _ if list.is_empty() => Strategy::Empty(self),
            Self::Vec(vec) if is_big(vec.len()) => Strategy::Big(Self::Vec(vec)),
            Self::PyIter(it) if it.bind(py).size_hint().0.pipe(is_big) => {
                Strategy::Big(Self::PyIter(it))
            }
            Self::PyIter(_) | Self::Vec(_) => Strategy::Small(self),
        }
    }
}
enum Strategy {
    Empty(IntoList),
    Small(IntoList),
    Big(IntoList),
}
