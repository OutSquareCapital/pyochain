use crate::{DictData, ListDataOwner, ListsDataMethods, types::ListOrAny};
use either::Either;
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
    let dict = mapping.get_dict().clone_ref(py).into_bound(py);
    try_cast! {
        match index {
            Case::PySlice(slice) => {
                let keys = mapping.list_mut().inner_mut().get_slice(slice)?;
                mapping.list_mut().del_slice(slice)?;
                for key in keys {
                    dict.del_item(key)?;
                }
                Ok(())
            },
            int => {
                let key = mapping.list_mut().pop(py, int.extract::<isize>()?)?;
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
    let dict = mapping.get_dict().clone_ref(py).into_bound(py).into_any();
    try_cast! {
        match index {
            Case::PySlice(slice) => mapping
                .list_mut()
                .inner_mut()
                .get_slice(slice)?
                .iter()
                .map(|key| tuple!(key.bind(py), &dict.get_item(key)?).map(Bound::into_any))
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
    let dict = mapping.get_dict().clone_ref(py).into_bound(py).into_any();
    try_cast! {
        match index {
            Case::PySlice(slice) => mapping
                .list_mut()
                .inner_mut()
                .get_slice(slice)?
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
            Case::PySlice(slice) => mapping
                .list_mut()
                .inner_mut()
                .get_slice(slice)?
                .iter()
                .collect_bound::<PyList>(py)
                .map(Either::Left),
            int => mapping.extract_index(&int).map(Either::Right),
        }
    }
}
