use std::cmp::Ordering;

use pyo3::prelude::*;
use pyo3_ext::pylibs;
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
    F: Fn(&Py<PyAny>, &Py<PyAny>) -> Ordering,
{
    let mut values = match data {
        ListAdd::Identity => list.collapse(py).tap_mut(|v| v.sort_by(&func)),
        ListAdd::Vec(vec) => vec.tap_mut(|v| v.sort_by(&func)),
        ListAdd::Sorted(sorted) => sorted.collapse(py).tap_mut(|v| v.sort_by(&func)),
        ListAdd::Iterator(it) => it
            .pipe(|x| pylibs::builtins::sorted(&x, false, py_func))?
            .iter()
            .map(Bound::unbind)
            .collect(),
    };
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
