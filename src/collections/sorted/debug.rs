use crate::collections::{
    SortedDict, SortedKeyDict, SortedKeyList, SortedList,
    sorted::{
        getters::ListGetter,
        set::{SortedKeySet, SortedSet},
    },
};
use pyo3::prelude::*;
use pyo3_ext::types::BoundedEither;
use sorted_rs::debug;
use tap::Pipe;

#[pyfunction]
pub fn check_sorted_dict(data: BoundedEither<'_, SortedDict, SortedKeyDict>) -> PyResult<()> {
    data.map_either(
        |x| debug::check_dict(x.py(), &x.get().lock()),
        |x| debug::check_dict(x.py(), &x.get().lock()),
    )
    .into_inner()
}

#[pyfunction]
pub fn check_sorted_set(data: BoundedEither<'_, SortedSet, SortedKeySet>) -> PyResult<()> {
    data.map_either(
        |x| debug::check_set_len(x.py(), &x.get().lock()),
        |x| debug::check_set_len(x.py(), &x.get().lock()),
    )
    .into_inner()
}

#[pyfunction]
pub fn assert_sorted_list_empty(lst: BoundedEither<'_, SortedList, SortedKeyList>) -> PyResult<()> {
    lst.map_either(
        |x| debug::check_empty(&x.get().lock()),
        |x| debug::check_empty(&x.get().lock()),
    )
    .into_inner()
}
#[pyfunction]
pub fn check_sorted_list(data: &Bound<'_, SortedList>) -> PyResult<()> {
    data.get().lock().pipe(|x| debug::check_list(data.py(), &x))
}
#[pyfunction]
pub fn check_sorted_key_list(data: &Bound<'_, SortedKeyList>) -> PyResult<()> {
    debug::check_key_list(data.py(), &data.get().lock())
}
