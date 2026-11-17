use std::cmp::Ordering;

use derive_more::Constructor;
use itertools::Itertools;
use parking_lot::RwLock;
use pyo3::{prelude::*, types::PyList};
use pyo3_ext::{prelude::*, types::PyFrozenClass};
use pyochain_macros::try_cast;
use tap::prelude::*;

use crate::{ops, traits::ListsMethods, types::VecPy};

pub trait PyListsDataRef: PyFrozenClass + AsRef<RwLock<Self::L>> + From<Self::L> {
    type L: ListsMethods;
    fn try_as_updatable<'py>(&self, right: &'py Bound<'py, PyAny>) -> PyResult<ListAdd<'py>> {
        let py = right.py();
        try_cast! {
            match right {
                CaseExact::Self(list) if std::ptr::eq(list.get(), self) => Ok(ListAdd::Identity),
                CaseExact::Self(list) => list
                    .get()
                    .as_ref()
                    .read()
                    .values
                    .iter_owned(py)
                    .collect_vec()
                    .into_iter()
                    .pipe(if Self::L::IS_KEYED {
                        ListAdd::Vec
                    } else {
                        ListAdd::Sorted
                    })
                    .pipe(Ok),
                CaseExact::PyList(list) => list.to_owned().pipe(ListAdd::PyListExt).pipe(Ok),
                // NOTE: Keeping the Iterator lazy here don't provide significant performance benefits,
                // NOTE: since `builtins::sorted` simply do list(iterable);list.sort() under the hood in CPython.
                _ => right.try_iter()?.try_collect_bound(py).map(ListAdd::PyListOwned),
            }
        }
    }
}
pub enum ListAdd<'py> {
    Identity,
    /// Values in no particular order.
    Vec(std::vec::IntoIter<Py<PyAny>>),
    /// Values already sorted according to the order of the list they are added to.
    Sorted(std::vec::IntoIter<Py<PyAny>>),
    /// A Python list externally owned by the caller. Mutating it could lead to undesired behavior.
    PyListExt(Bound<'py, PyList>),
    /// A Python list created by internal computations, meaning it don't have to take into account mutability issues.
    PyListOwned(Bound<'py, PyList>),
}
impl<'py> ListAdd<'py> {
    pub fn into_new_list<T: ListsMethods>(self, py: Python<'_>, list: &T) -> PyResult<T> {
        let mut new_inst = list.new_empty(py);
        self.extend_into(py, &mut new_inst)?;
        Ok(new_inst)
    }
    pub fn concat_into<T: ListsMethods>(self, py: Python<'_>, list: &T) -> PyResult<T> {
        let mut new_inst = list.new_empty(py);
        let key = list.key_func(py);
        let slf = list.values.iter_owned(py);
        let func = |a: &Py<PyAny>, b: &Py<PyAny>| key.cmp(a, b);
        match self {
            ListAdd::Identity => slf.chain(list.values.iter_owned(py)).sorted_by(func),
            ListAdd::PyListExt(pylist) | ListAdd::PyListOwned(pylist) => slf
                .chain(pylist.into_iter().map(Bound::unbind))
                .sorted_by(func),
            ListAdd::Vec(vec) | ListAdd::Sorted(vec) => slf.chain(vec).sorted_by(func),
        }
        .pipe(|it| update_chunked(it, &mut new_inst, &key))?;
        Ok(new_inst)
    }
    #[inline]
    pub fn extend_into(self, py: Python<'py>, list: &mut impl ListsMethods) -> PyResult<()> {
        //TODO: Check `itertools::merge` to see if it can be beneficial here.
        let key = list.key_func(py);
        match (ops::Update::new(list, &self), self) {
            (_, ListAdd::Identity) => list
                .values
                .iter_owned(py)
                .chain(list.values.iter_owned(py))
                .sorted_by(|a: &Py<PyAny>, b: &Py<PyAny>| key.cmp(a, b))
                .pipe(|it| update_chunked(it, list, &key)),
            (ops::Update::Empty, Self::Sorted(vec)) => {
                vec.pipe(|it| update_chunked(it, list, &key))
            }
            (ops::Update::Empty, Self::Vec(vec)) => vec
                .collect_bound::<PyList>(py)?
                .pipe(|pylist| list_into_update(pylist, list, &key)),
            (ops::Update::Empty, Self::PyListExt(pylist)) => pylist
                .copy()?
                .pipe(|pylist| list_into_update(pylist, list, &key)),
            (ops::Update::Empty, Self::PyListOwned(pylist)) => list_into_update(pylist, list, &key),
            (ops::Update::Big, Self::Vec(vec) | Self::Sorted(vec)) => list
                .values
                .iter_owned(py)
                .chain(vec)
                .collect_bound::<PyList>(py)?
                .pipe(|pylist| list_into_update(pylist, list, &key)),
            (ops::Update::Big, ListAdd::PyListExt(pylist) | ListAdd::PyListOwned(pylist)) => list
                .values
                .iter_owned(py)
                .chain(pylist.iter().map(Bound::unbind))
                .collect_bound::<PyList>(py)?
                .pipe(|pylist| list_into_update(pylist, list, &key)),
            (ops::Update::Small, Self::Vec(vec) | Self::Sorted(vec)) => vec
                .map(|x| x.into_bound(py))
                .try_for_each(|val| list.append(val)),
            (ops::Update::Small, ListAdd::PyListExt(pylist) | ListAdd::PyListOwned(pylist)) => {
                pylist.into_iter().try_for_each(|val| list.append(val))
            }
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Vec(it) | Self::Sorted(it) => it.len(),
            ListAdd::PyListExt(pylist) | ListAdd::PyListOwned(pylist) => pylist.len(),
            Self::Identity => 0,
        }
    }
}
fn list_into_update<'py>(
    pylist: Bound<'py, PyList>,
    list: &mut impl ListsMethods,
    key: &impl KeyFn<'py>,
) -> PyResult<()> {
    pylist
        .py_sorted_by(key)?
        .into_iter()
        .map(Bound::unbind)
        .pipe(|it| update_chunked(it, list, key))
}
#[inline]
fn update_chunked<'py>(
    it: impl ExactSizeIterator<Item = Py<PyAny>>,
    list: &mut impl ListsMethods,
    key: &impl KeyFn<'py>,
) -> PyResult<()> {
    list.clear(key.py());
    list.len = it.len();
    it.chunked(*list.load)
        .try_for_each(|chunk| key.push_chunk(list, chunk))
}

/// Strategy for ordering and storing the elements added to a sorted list.\
/// [`Identity`] sorts by value, [`WithKey`] sorts by the result of a Python callable.
pub trait KeyFn<'py> {
    fn py(&self) -> Python<'py>;
    fn cmp(&self, a: &Py<PyAny>, b: &Py<PyAny>) -> Ordering;
    fn sort_pylist(&self, pylist: &Bound<'py, PyList>) -> PyResult<()>;
    fn push_chunk(&self, list: &mut impl ListsMethods, vals: VecPy) -> PyResult<()>;
}
pub struct Identity<'py>(pub Python<'py>);
pub struct WithKey<'py>(pub Bound<'py, PyAny>);
impl<'py> KeyFn<'py> for Identity<'py> {
    fn py(&self) -> Python<'py> {
        self.0
    }
    fn cmp(&self, a: &Py<PyAny>, b: &Py<PyAny>) -> Ordering {
        a.bind(self.0).compare_fast(b.bind(self.0)).unwrap()
    }
    fn sort_pylist(&self, pylist: &Bound<'py, PyList>) -> PyResult<()> {
        pylist.sort()
    }
    fn push_chunk(&self, list: &mut impl ListsMethods, vals: VecPy) -> PyResult<()> {
        list.push_chunk(self.0, vals);
        Ok(())
    }
}
impl<'py> KeyFn<'py> for WithKey<'py> {
    fn py(&self) -> Python<'py> {
        self.0.py()
    }
    fn cmp(&self, a: &Py<PyAny>, b: &Py<PyAny>) -> Ordering {
        self.0
            .call1((a,))
            .unwrap()
            .compare_fast(&self.0.call1((b,)).unwrap())
            .unwrap()
    }
    fn sort_pylist(&self, pylist: &Bound<'py, PyList>) -> PyResult<()> {
        pylist.sort_by(&self.0, false)
    }
    fn push_chunk(&self, list: &mut impl ListsMethods, vals: VecPy) -> PyResult<()> {
        let py = self.py();
        vals.iter()
            .map(|v| self.0.call1((v,)))
            .map_ok(Bound::unbind)
            .collect::<PyResult<VecPy>>()
            .map(|key_chunk| {
                list.push_chunk(py, key_chunk);
                list.values.push(vals);
            })
    }
}
trait PyListSort<'py>: Sized {
    fn py_sorted_by(self, key: &impl KeyFn<'py>) -> PyResult<Bound<'py, PyList>>;
}
impl<'py> PyListSort<'py> for Bound<'py, PyList> {
    fn py_sorted_by(self, key: &impl KeyFn<'py>) -> PyResult<Self> {
        key.sort_pylist(&self)?;
        Ok(self)
    }
}
trait IntoChunkedIter<I> {
    fn chunked(self, load: usize) -> ChunkedIter<I>;
}
impl<I: ExactSizeIterator> IntoChunkedIter<I> for I {
    fn chunked(self, load: usize) -> ChunkedIter<I> {
        self.into_iter().pipe(|iter| ChunkedIter::new(iter, load))
    }
}
#[derive(Debug, Constructor)]
struct ChunkedIter<I> {
    iterator: I,
    load: usize,
}
impl<I, T> Iterator for ChunkedIter<I>
where
    I: ExactSizeIterator<Item = T>,
{
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iterator
            .by_ref()
            .take(self.load)
            .collect::<Vec<_>>()
            .pipe(|chunk| (!chunk.is_empty()).then_some(chunk))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let chunks = self.iterator.len().div_ceil(self.load);
        (chunks, Some(chunks))
    }
}
