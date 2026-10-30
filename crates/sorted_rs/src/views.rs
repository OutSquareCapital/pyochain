use crate::{DictData, prelude::*, slice::SliceIdxs, types::ListOrAny};
use either::Either;
use itertools::Itertools;
use pyo3::{
    prelude::*,
    types::{PyList, PySlice},
};
use pyo3_ext::{prelude::CollectBoundIterator, tuple};
use pyochain_macros::try_cast;

pub fn delitem<T: ListsDataMethods>(
    mapping: &mut DictData<T>,
    index: Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = index.py();
    let dict = mapping.1.clone_ref(py).into_bound(py);
    try_cast! {
        match index {
            Case::PySlice(pyslice) => {
                let slice = SliceIdxs::new(pyslice, mapping.len)?;
                let keys = slice.extract_from(mapping)?;
                slice.delete_from(&mut mapping.0)?;
                for key in keys {
                    dict.del_item(key)?;
                }
                Ok(())
            }
            int => {
                let key = mapping.0.pop(py, int.extract::<isize>()?)?;
                dict.del_item(key)?;
                Ok(())
            }
        }
    }
}
#[inline(always)]
pub fn get_item_for_items<'py, T: ListsDataMethods>(
    mapping: &mut DictData<T>,
    index: Bound<'py, PyAny>,
) -> PyResult<ListOrAny<'py>> {
    let py = index.py();
    let dict = mapping.1.clone_ref(py).into_bound(py).into_any();
    try_cast! {
        match index {
            Case::PySlice(slice) => SliceIdxs::new(slice, mapping.len)?
                .extract_from(mapping)?
                .iter()
                .map(|key| tuple!(key.bind(py), &dict.get_item(key)?))
                .map_ok(Bound::into_any)
                .try_collect_bound::<PyList>(py)
                .map(Either::Left),
            int => {
                let key = mapping.extract_index(&int)?;
                let value = dict.get_item(&key)?;
                tuple!(key, value).map(Bound::into_any).map(Either::Right)
            }
        }
    }
}
#[inline(always)]
pub fn get_item_for_values<'py, T: ListsDataMethods>(
    mapping: &mut DictData<T>,
    index: Bound<'py, PyAny>,
) -> PyResult<ListOrAny<'py>> {
    let py = index.py();
    let dict = mapping.1.clone_ref(py).into_bound(py).into_any();
    try_cast! {
        match index {
            Case::PySlice(slice) => SliceIdxs::new(slice, mapping.len)?
                .extract_from(mapping)?
                .iter()
                .map(|key| dict.get_item(key))
                .try_collect_bound::<PyList>(py)
                .map(Either::Left),
            int => dict
                .get_item(mapping.extract_index(&int)?)
                .map(Either::Right),
        }
    }
}
#[inline(always)]
pub fn get_item_for_keys<'py, T: ListsDataMethods>(
    mapping: &mut DictData<T>,
    index: Bound<'py, PyAny>,
) -> PyResult<ListOrAny<'py>> {
    let py = index.py();
    try_cast! {
        match index {
            Case::PySlice(slice) => SliceIdxs::new(slice, mapping.len)?
                .extract_from(mapping)?
                .iter()
                .collect_bound::<PyList>(py)
                .map(Either::Left),
            int => mapping.extract_index(&int).map(Either::Right),
        }
    }
}
