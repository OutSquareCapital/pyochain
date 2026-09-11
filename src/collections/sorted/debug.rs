use crate::collections::{
    SortedDict, SortedKeyDict, SortedKeyList, SortedList,
    sorted::{
        set::{SortedKeySet, SortedSet},
        traits::{ListGetter, SortedDictMethods, SortedSetMethods},
    },
};
use either::Either;
use pyo3::prelude::*;
use sorted_rs::{InnerGetter, ListDataGetters, debug::check_key_list, pyassert};

#[pyfunction]
pub fn check_sorted_dict(
    py: Python<'_>,
    data: Either<Py<SortedDict>, Py<SortedKeyDict>>,
) -> PyResult<()> {
    data.map_either(|x| check_dict(x.get(), py), |x| check_dict(x.get(), py))
        .into_inner()
}

fn check_dict(x: &impl SortedDictMethods, py: Python<'_>) -> PyResult<()> {
    let data = x.try_lock();
    data.inner().check(py)?;

    pyassert!(x.len(py) == data.len());
    pyassert!(data.inner().iter().all(|item| {
        x.contains(item.bind(py))
            .expect("Failed to check dict membership")
    }));
    Ok(())
}
#[pyfunction]
pub fn check_sorted_set(
    py: Python<'_>,
    data: Either<Py<SortedSet>, Py<SortedKeySet>>,
) -> PyResult<()> {
    data.map_either(
        |x| check_set_len(x.get(), py),
        |x| check_set_len(x.get(), py),
    )
    .into_inner()
}

fn check_set_len<T: SortedSetMethods>(checked: &T, py: Python<'_>) -> PyResult<()> {
    let set = checked.get_set(py);
    let data = checked.try_lock();
    pyassert!(set.len() == data.len());
    data.inner().check(py)?;
    pyassert!(
        data.inner()
            .iter()
            .all(|x| set.contains(x).expect("Failed to check set membership"))
    );
    Ok(())
}
#[pyfunction]
pub fn assert_sorted_list_empty(lst: Either<Py<SortedList>, Py<SortedKeyList>>) -> PyResult<()> {
    match lst {
        Either::Left(x) => x.get().try_lock().inner().check_empty(),
        Either::Right(x) => x.get().try_lock().inner().check_empty(),
    }
}
#[pyfunction]
pub fn check_sorted_list(py: Python<'_>, data: &Bound<'_, SortedList>) -> PyResult<()> {
    data.get().try_lock().inner().check(py)
}
#[pyfunction]
pub fn check_sorted_key_list(py: Python<'_>, data: &Bound<'_, SortedKeyList>) -> PyResult<()> {
    check_key_list(py, &data.get().try_lock())
}
