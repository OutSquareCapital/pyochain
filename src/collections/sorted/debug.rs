use crate::collections::{
    SortedDict, SortedKeyDict, SortedKeyList, SortedList,
    sorted::{
        set::{SortedKeySet, SortedSet},
        traits::ListGetter,
    },
};
use either::Either;
use pyo3::prelude::*;
use pyo3_ext::types::BoundedEither;
use sorted_rs::{InnerGetter, debug};
use tap::Pipe;

#[pyfunction]
pub fn check_sorted_dict(data: BoundedEither<'_, SortedDict, SortedKeyDict>) -> PyResult<()> {
    let py = data.py();
    data.map_either(
        |x| debug::check_dict(&x.get().lock(), py),
        |x| debug::check_dict(&x.get().lock(), py),
    )
    .into_inner()
}

#[pyfunction]
pub fn check_sorted_set(data: BoundedEither<'_, SortedSet, SortedKeySet>) -> PyResult<()> {
    let py = data.py();
    data.map_either(
        |x| debug::check_set_len(&x.get().lock(), py),
        |x| debug::check_set_len(&x.get().lock(), py),
    )
    .into_inner()
}

#[pyfunction]
pub fn assert_sorted_list_empty(lst: BoundedEither<'_, SortedList, SortedKeyList>) -> PyResult<()> {
    match lst {
        Either::Left(x) => x.get().lock().inner().pipe(debug::check_empty),
        Either::Right(x) => x.get().lock().inner().pipe(debug::check_empty),
    }
}
#[pyfunction]
pub fn check_sorted_list(data: &Bound<'_, SortedList>) -> PyResult<()> {
    data.get()
        .lock()
        .inner()
        .pipe(|x| debug::check_list(data.py(), x))
}
#[pyfunction]
pub fn check_sorted_key_list(data: &Bound<'_, SortedKeyList>) -> PyResult<()> {
    debug::check_key_list(data.py(), &data.get().lock())
}
