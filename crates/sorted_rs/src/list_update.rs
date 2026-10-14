use std::cmp::Ordering;

use itertools::Itertools;
use pyo3::{prelude::*, types::PyList};
use pyo3_ext::{
    prelude::{CollectBoundIterator, PyListExtMethods},
    pylibs,
};
use tap::prelude::*;

use crate::{ListAdd, traits::ListsDataMethods, types::VecPy};
#[inline]
pub fn dispatch<T, F>(
    list: &mut T,
    py: Python<'_>,
    data: ListAdd<'_, T>,
    func: F,
    py_func: Option<&Bound<'_, PyAny>>,
) -> PyResult<()>
where
    T: ListsDataMethods,
    F: FnMut(&Py<PyAny>, &Py<PyAny>) -> Ordering,
{
    //TODO: Find a way to avoid unecessary intermediate Vecs by flattening `ListAdd` and length checking paths branches,
    // with precise cases instead of two sequential steps,
    // without bloating the code.
    //TODO: Check `itertools::merge` to see if it can be beneficial here.
    let values = match data {
        ListAdd::Identity => list.iter_owned(py).sorted_by(func).collect(),
        ListAdd::Vec(vec) => vec.tap_mut(|v| v.sort_by(func)),
        ListAdd::Sorted(sorted) => sorted.iter_owned(py).sorted_by(func).collect(),
        ListAdd::Iterator(it) => pylibs::builtins::sorted(&it, false, py_func)?
            .iter()
            .map(Bound::unbind)
            .collect(),
    };
    if list.maxes.is_empty() {
        update_chunked(list, py, values.into_iter())
    } else if 4 * values.len() >= list.len {
        list.iter()
            .chain(values.iter())
            .collect_bound::<PyList>(py)
            .and_then(|pylist| {
                py_func
                    .map_or_else(|| pylist.sort(), |key| pylist.sort_by(key, false))
                    .map(|()| pylist)
            })?
            .iter()
            .map(Bound::unbind)
            .pipe(|pylist| update_chunked(list, py, pylist))
    } else {
        values
            .into_iter()
            .map(|x| x.into_bound(py))
            .try_for_each(|val| list.add(val))
    }
}
#[inline]
fn update_chunked<T, I>(list: &mut T, py: Python<'_>, mut it: I) -> PyResult<()>
where
    T: ListsDataMethods,
    I: ExactSizeIterator<Item = Py<PyAny>>,
{
    list.clear(py);
    list.len = it.len();
    let load = list.load;
    std::iter::from_fn(|| {
        it.by_ref()
            .take(load)
            .collect::<VecPy>()
            .pipe(|chunk| (!chunk.is_empty()).then_some(chunk))
    })
    .pipe(|chunked| list.finalize_update(py, chunked))
}
