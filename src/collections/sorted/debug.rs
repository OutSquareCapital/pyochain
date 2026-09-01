use crate::collections::{
    SortedDict, SortedKeyDict, SortedKeyList, SortedList,
    sorted::{
        set::{SortedKeySet, SortedSet},
        traits::{BaseSortedDict, BaseSortedSet, ListGetter},
    },
};
use either::Either;
use pyo3::prelude::*;
use sorted_rs::{ListDataGetters, ListsDataMethods, debug::check_key_list, pyassert};

#[pyfunction]
pub fn check_sorted_dict(
    py: Python<'_>,
    data: Either<Py<SortedDict>, Py<SortedKeyDict>>,
) -> PyResult<()> {
    data.map_either(|x| check_dict(x.get(), py), |x| check_dict(x.get(), py))
        .into_inner()
}

fn check_dict(x: &impl BaseSortedDict, py: Python<'_>) -> PyResult<()> {
    let data = x.get_data();
    data.check(py)?;

    pyassert!(x.len(py) == data.length());
    pyassert!(data.iter().all(|item| {
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

fn check_set_len<T: BaseSortedSet>(checked: &T, py: Python<'_>) -> PyResult<()> {
    let set = checked.get_set().clone_ref(py).into_bound(py);
    let data = checked.get_data();
    pyassert!(set.len() == data.length());
    data.check(py)?;
    pyassert!(
        data.iter()
            .all(|x| set.contains(x).expect("Failed to check set membership"))
    );
    Ok(())
}
#[pyfunction]
pub fn assert_sorted_list_empty(lst: Either<Py<SortedList>, Py<SortedKeyList>>) -> PyResult<()> {
    match lst {
        Either::Left(x) => x.get().get_data().check_empty(),
        Either::Right(x) => x.get().get_data().check_empty(),
    }
}
#[pyfunction]
pub fn check_sorted_list(py: Python<'_>, data: &Bound<'_, SortedList>) -> PyResult<()> {
    data.get().get_data().check(py)
}
#[pyfunction]
pub fn check_sorted_key_list(py: Python<'_>, data: &Bound<'_, SortedKeyList>) -> PyResult<()> {
    check_key_list(py, &data.get().get_data())
}
