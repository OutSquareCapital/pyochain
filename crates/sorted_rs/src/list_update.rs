use std::cmp::Ordering;

use derive_more::From;
use itertools::Itertools;
use parking_lot::{RwLock, RwLockReadGuard};
use pyo3::{
    prelude::*,
    types::{PyIterator, PyList},
};
use pyo3_ext::{
    prelude::{CollectBoundIterator, PyListExtMethods},
    pylibs,
    types::PyFrozenClass,
};
use tap::prelude::*;

use crate::{traits::ListsDataMethods, types::VecPy};

pub trait PyListsDataRef: PyFrozenClass + AsRef<RwLock<Self::L>> + From<Self::L> {
    type L: ListsDataMethods;
    fn try_as_updatable<'py>(
        &self,
        right: &'py Bound<'py, PyAny>,
    ) -> PyResult<ListAdd<'py, Self::L>> {
        match right.cast_exact::<Self>().map(Bound::get) {
            Ok(list) if std::ptr::eq(list, self) => Ok(ListAdd::Identity),
            Ok(list) => list.as_ref().read().pipe(ListAdd::Sorted).pipe(Ok),
            Err(_) => right.try_iter().map(ListAdd::Iterator),
        }
    }
}
#[derive(From)]
pub enum ListAdd<'py, T> {
    Identity,
    Sorted(RwLockReadGuard<'py, T>),
    Vec(VecPy),
    Iterator(Bound<'py, PyIterator>),
}
impl<T: ListsDataMethods> ListAdd<'_, T> {
    #[inline]
    pub fn merge_with(
        self,
        list: &mut T,
        py: Python<'_>,
        func: impl FnMut(&Py<PyAny>, &Py<PyAny>) -> Ordering,
        py_func: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        //TODO: Find a way to avoid unecessary intermediate Vecs by flattening `ListAdd` and length checking paths branches,
        // with precise cases instead of two sequential steps,
        // without bloating the code.
        //TODO: Check `itertools::merge` to see if it can be beneficial here.
        let values = match self {
            Self::Identity => list.iter_owned(py).sorted_by(func).collect(),
            Self::Vec(vec) => vec.tap_mut(|v| v.sort_by(func)),
            Self::Sorted(sorted) => sorted.iter_owned(py).sorted_by(func).collect(),
            Self::Iterator(it) => pylibs::builtins::sorted(&it, false, py_func)?
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
                .try_for_each(|val| list.append(val))
        }
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
