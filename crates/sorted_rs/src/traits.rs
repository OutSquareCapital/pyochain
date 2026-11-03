use std::ops::{Deref, DerefMut};

use crate::{
    Bounds, Loc, RangeIdxs,
    bisect::Bisect,
    bounds::IrangeSpecs,
    errors,
    inner::InnerData,
    list_update::ListAdd,
    reprs::PyRepr,
    slice::SliceIdxs,
    types::{IntOrSlice, ListOrAny, VecPy},
};
use either::Either;
use itertools::Itertools;
use pyo3::{prelude::*, types::PyList};
use pyo3_ext::prelude::*;
use tap::prelude::*;
pub(super) trait NestedVec {
    fn loc(&self, loc: &Loc) -> &Py<PyAny>;
    fn loc_insert(&mut self, loc: &Loc, value: Py<PyAny>);
    fn loc_remove(&mut self, loc: &Loc);
    fn loc_push(&mut self, loc: &Loc, value: Py<PyAny>);
    fn loc_last(&self, loc: &Loc) -> &Py<PyAny>;
    fn loc_len(&self, loc: &Loc) -> usize;
}
impl NestedVec for [Vec<Py<PyAny>>] {
    fn loc(&self, loc: &Loc) -> &Py<PyAny> {
        &self[loc.pos][loc.idx]
    }
    fn loc_insert(&mut self, loc: &Loc, value: Py<PyAny>) {
        self[loc.pos].insert(loc.idx, value);
    }
    fn loc_remove(&mut self, loc: &Loc) {
        self[loc.pos].remove(loc.idx);
    }
    fn loc_push(&mut self, loc: &Loc, value: Py<PyAny>) {
        self[loc.pos].push(value);
    }
    fn loc_last(&self, loc: &Loc) -> &Py<PyAny> {
        self[loc.pos].last().unwrap()
    }
    fn loc_len(&self, loc: &Loc) -> usize {
        self[loc.pos].len()
    }
}

pub trait ListsDataMethods:
    Deref<Target = InnerData> + DerefMut + PyRepr + Sized + Send + Sync
{
    /// Returns a reference to the "keys" of the collection.\
    /// For collections sorted by *values*, this is equivalent to `InnerData::values`.\
    /// For collections sorted by *key*, this is equivalent to `KeysListsData::1`.
    fn keys(&self) -> &[VecPy];
    fn irange_specs(&self, specs: IrangeSpecs<'_>) -> PyResult<Bounds>;
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self>;
    /// Equivalent to `bisect_ky_key` if the collection is sorted **by value**.\
    /// If the collection is sorted **by key**, *value* is first transformed by the *key* function before the bisect operation.
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn expand(&mut self, py: Python<'_>, pos: usize);
    fn clear(&mut self, py: Python<'_>);
    fn delete(&mut self, py: Python<'_>, loc: Loc) -> PyResult<()>;
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>>;
    fn finalize_update(&mut self, py: Python<'_>, it: impl Iterator<Item = VecPy>) -> PyResult<()>;
    fn extend(&mut self, py: Python<'_>, values: ListAdd<'_, Self>) -> PyResult<()>;
    fn index(
        &mut self,
        value: &Bound<'_, PyAny>,
        indices: RangeIdxs<Option<isize>>,
    ) -> PyResult<usize>;
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn bisect_by_key<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        if self.maxes.is_empty() {
            Ok(0)
        } else {
            let mut loc = Loc::new(0, 0);
            loc.pos = self.maxes.bisect::<LEFT>(value)?;
            if loc.pos == self.maxes.len() {
                Ok(self.len)
            } else {
                loc.idx = self.keys()[loc.pos].bisect::<LEFT>(value)?;
                Ok(self.loc(&loc))
            }
        }
    }
    #[inline]
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.find(value).map(|x| x.is_some())
    }
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        self.as_owned_from(py, self.collapse(py))
    }
    fn concat(&self, py: Python<'_>, other: ListAdd<'_, Self>) -> PyResult<Self> {
        let out = match other {
            ListAdd::Identity => self.repeat(py, 2),
            ListAdd::Sorted(list) => self
                .iter()
                .chain(list.iter())
                .map(|x| x.clone_ref(py))
                .collect(),
            ListAdd::Vec(values) => values,
            ListAdd::Iterator(it) => self
                .iter()
                .map(|x| x.clone_ref(py).pipe(Ok::<Py<PyAny>, PyErr>))
                .chain(it.map_ok(Bound::unbind))
                .collect::<PyResult<_>>()?,
        };
        self.as_owned_from(py, out)
    }
    fn get_item_or_slice<'py>(&mut self, index: IntOrSlice<'py>) -> PyResult<ListOrAny<'py>> {
        match index {
            Either::Right(slice) => SliceIdxs::new(&slice, self.len)?
                .extract_from(slice.py(), self)?
                .iter()
                .collect_bound::<PyList>(slice.py())
                .map(Either::Left),
            Either::Left(index) => self
                .get_item(index.py(), index.extract()?)
                .map(Either::Right),
        }
    }
    fn del_item_or_slice(&mut self, index: IntOrSlice<'_>) -> PyResult<()> {
        match index {
            Either::Right(slice) => SliceIdxs::new(&slice, self.len)?.delete_from(slice.py(), self),
            Either::Left(index) => self.del_item(index.py(), index.extract()?),
        }
    }
    fn del_item(&mut self, py: Python<'_>, index: isize) -> PyResult<()> {
        if self.len == 0 {
            Err(errors::out_of_range())
        } else {
            let loc = self.set_pos(index)?;
            self.delete(py, loc)
        }
    }
    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.find(&value)?
            .map_or(Ok(()), |loc| self.delete(value.py(), loc))
    }
    fn imul(&mut self, py: Python<'_>, num: usize) -> PyResult<()> {
        let values = self.repeat(py, num);
        self.clear(py);
        self.extend(py, values.into())
    }
    fn pop<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let loc = self.index_to_loc(index)?;
        let val = self.values.loc(&loc).clone_ref(py);
        self.delete(py, loc)?;
        Ok(val.into_bound(py))
    }
    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.find(value)? {
            Some(loc) => self.delete(value.py(), loc),
            None => Err(errors::not_in_list(value)),
        }
    }
    fn as_repeated(&self, py: Python<'_>, num: usize) -> PyResult<Self> {
        self.as_owned_from(py, self.repeat(py, num))
    }
    fn reset(&mut self, py: Python<'_>, load: usize) -> PyResult<()> {
        let values = self.collapse(py);
        self.clear(py);
        self.load = load;
        self.extend(py, values.into())
    }
    fn set_item(&mut self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.del_item(value.py(), index)?;
        self.append(value)
    }
}
