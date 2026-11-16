use crate::collections::{
    SortedDict, SortedKeyDict, SortedKeyList, SortedList,
    sorted::set::{SortedKeySet, SortedSet},
};
use pyo3::prelude::*;
use pyo3_ext::types::BoundedEither;
use sorted_rs::{debug, prelude::*};

#[pyfunction]
pub fn check_sorted_dict(data: BoundedEither<'_, SortedDict, SortedKeyDict>) -> PyResult<()> {
    either::for_both!(data, x => debug::check_dict(x.py(), &x.get().read()))
}
#[pyfunction]
pub fn check_sorted_set(data: BoundedEither<'_, SortedSet, SortedKeySet>) -> PyResult<()> {
    either::for_both!(data, x => debug::check_set(x.py(), &x.get().read()))
}
#[pyfunction]
pub fn assert_sorted_list_empty(lst: BoundedEither<'_, SortedList, SortedKeyList>) -> PyResult<()> {
    either::for_both!(lst, x => debug::check_empty(&x.get().read()))
}
#[pyfunction]
pub fn check_sorted_list(data: &Bound<'_, SortedList>) -> PyResult<()> {
    debug::check_list(data.py(), data.get().read().as_ref())
}
#[pyfunction]
pub fn check_sorted_key_list(data: &Bound<'_, SortedKeyList>) -> PyResult<()> {
    debug::check_key_list(data.py(), &data.get().read())
}
