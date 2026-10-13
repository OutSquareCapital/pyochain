use std::cmp::Ordering;

use pyo3::prelude::*;
use tap::prelude::*;

use crate::{traits::ListsDataMethods, types::VecPy};
#[inline]
pub fn dispatch<T, F>(list: &mut T, py: Python<'_>, mut values: VecPy, func: F) -> PyResult<()>
where
    T: ListsDataMethods,
    F: Fn(&Py<PyAny>, &Py<PyAny>) -> Ordering,
{
    values.sort_by(&func);
    if list.maxes.is_empty() {
        update_chunked(list, py, values)
    } else if 4 * values.len() >= list.len {
        list.values.push(values);
        values = list.collapse(py);
        values.sort_by(func);
        update_chunked(list, py, values)
    } else {
        values
            .into_iter()
            .map(|x| x.into_bound(py))
            .try_for_each(|val| list.add(val))
    }
}
#[inline]
fn update_chunked<T>(list: &mut T, py: Python<'_>, values: VecPy) -> PyResult<()>
where
    T: ListsDataMethods,
{
    list.clear(py);
    list.len = values.len();
    let load = list.load;
    let mut it = values.into_iter();
    std::iter::from_fn(|| {
        it.by_ref()
            .take(load)
            .collect::<VecPy>()
            .pipe(|chunk| (!chunk.is_empty()).then_some(chunk))
    })
    .pipe(|chunked| list.finalize_update(py, chunked))
}
