use derive_more::Constructor;
use either::Either;
use pyo3::{prelude::*, types::PySet};
use pyo3_ext::prelude::*;

use crate::{
    slice::SliceIdxs,
    traits::ListsMethods,
    types::{IndexOut, IntoIndex, Set},
    views::ViewMethods,
};
#[derive(Constructor)]
pub struct Index<'py>(IntoIndex<'py>);
impl<'py> Index<'py> {
    #[inline]
    pub fn get_from_list<T: ListsMethods>(self, list: &mut T) -> PyResult<IndexOut<'py>> {
        let py = self.0.py();
        match self.0 {
            Either::Left(slice) => SliceIdxs::new(&slice, list.len())?
                .extract_from(py, list)?
                .iter()
                .collect_bound(py)
                .map(Either::Left),
            Either::Right(index) => list.get_item(py, index).map(Either::Right),
        }
    }
    #[inline]
    pub fn get_from_values_view(self, view: &impl ViewMethods) -> PyResult<IndexOut<'py>> {
        let py = self.0.py();
        let mut mapping = view.mapping_mut();
        let dict = mapping.data.clone_ref(py).into_bound(py).into_any();
        match self.0 {
            Either::Left(slice) => SliceIdxs::new(&slice, mapping.len())?
                .extract_from(py, &mut mapping.list)?
                .iter()
                .map(|key| dict.get_item(key))
                .try_collect_bound(py)
                .map(Either::Left),
            Either::Right(int) => dict
                .get_item(mapping.list.get_item(py, int)?)
                .map(Either::Right),
        }
    }
    #[inline]
    pub fn get_from_items_view(self, view: &impl ViewMethods) -> PyResult<IndexOut<'py>> {
        let py = self.0.py();
        let mut mapping = view.mapping_mut();
        let dict = mapping.data.clone_ref(py).into_bound(py).into_any();
        match self.0 {
            Either::Left(slice) => SliceIdxs::new(&slice, mapping.len())?
                .extract_from(py, &mut mapping.list)?
                .iter()
                .map(|key| tuple!(key.bind(py), &dict.get_item(key)?))
                .try_collect_bound(py)
                .map(Either::Left),
            Either::Right(int) => {
                let key = mapping.list.get_item(py, int)?;
                let value = dict.get_item(&key)?;
                tuple!(key, value).map(Bound::into_any).map(Either::Right)
            }
        }
    }
    #[inline]
    pub fn del_from_list<T: ListsMethods>(self, list: &mut T) -> PyResult<()> {
        let py = self.0.py();
        match self.0 {
            Either::Left(slice) => SliceIdxs::new(&slice, list.len())?.delete_from(py, list),
            Either::Right(index) => list.del_item(py, index),
        }
    }
    #[inline]
    pub fn del_from_set<T: ListsMethods>(self, set: &mut Set<T>) -> PyResult<()> {
        let py = self.0.py();
        match self.0 {
            Either::Left(pyslice) => {
                let slice = SliceIdxs::new(&pyslice, set.len())?;
                let values = slice
                    .extract_from(py, &mut set.list)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                set.data.bind(py).difference_update((values,))?;
                slice.delete_from(py, &mut set.list)
            }
            Either::Right(int) => {
                let value = set.list.get_item(py, int)?;
                set.data.bind(py).remove(&value)?;
                set.list.del_item(py, int)
            }
        }
    }
    #[inline]
    pub fn del_from_view<T: ViewMethods>(self, view: &T) -> PyResult<()> {
        let py = self.0.py();
        let mut mapping = view.mapping_mut();
        let dict = mapping.data.clone_ref(py).into_bound(py);
        match self.0 {
            Either::Left(pyslice) => {
                let slice = SliceIdxs::new(&pyslice, mapping.len())?;
                let keys = slice.extract_from(py, &mut mapping.list)?;
                slice.delete_from(py, &mut mapping.list)?;
                keys.iter().try_for_each(|key| dict.del_item(key))
            }
            Either::Right(int) => mapping.list.pop(py, int).and_then(|key| dict.del_item(key)),
        }
    }
}
