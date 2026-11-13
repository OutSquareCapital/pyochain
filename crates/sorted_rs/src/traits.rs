use std::ops::{Deref, DerefMut};

use crate::{
    bounds::{Bounds, Loc, RawIrangeSpecs},
    errors,
    inner::{InnerList, Values},
    iter::{IrangeBounds, builder::RawRangeIdxs},
    list_update::ListAdd,
    ops,
    reprs::PyRepr,
    slice::SliceIdxs,
    types::{IntOrSlice, ListOrAny, VecPy},
};
use either::Either;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3_ext::prelude::*;
use tap::prelude::*;

pub type BisectRes = PyResult<ops::Bisect>;
pub trait ListsMethods:
    Deref<Target = InnerList> + DerefMut + PyRepr + Sized + Send + Sync
{
    /// Returns a reference to the "keys" of the collection.\
    /// For collections sorted by *values*, this is equivalent to `InnerData::values`.\
    /// For collections sorted by *key*, this is equivalent to `KeysListsData::1`.
    fn keys(&self) -> &Values;
    fn keys_mut(&mut self) -> &mut Values;
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self>;
    /// Equivalent to `bisect_ky_key` if the collection is sorted **by value**.\
    /// If the collection is sorted **by key**, *value* is first transformed by the *key* function before the bisect operation.
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn expand_splitted(&mut self, py: Python<'_>, pos: usize) -> (VecPy, Py<PyAny>);
    fn clear(&mut self, py: Python<'_>);
    fn delete(&mut self, py: Python<'_>, loc: Loc) -> PyResult<()>;
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>>;
    fn extend(&mut self, py: Python<'_>, values: ListAdd<'_, Self>) -> PyResult<()>;
    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize>;
    fn irange<'py>(
        &self,
        py: Python<'py>,
        specs: RawIrangeSpecs<'py>,
    ) -> PyResult<IrangeBounds<'py>>;
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn delete_on_len_sup_to_half_load(&mut self, py: Python<'_>, loc: Loc) {
        self.maxes[loc.pos] = self.keys().loc_last(&loc).clone_ref(py);
        if self.lengths.is_built() {
            self.lengths.delete_at(loc.pos);
        }
    }
    #[inline]
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.find(value).map(|x| x.is_some())
    }
    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        self.as_owned_from(py, self.values.iter_owned(py).collect_vec())
    }
    fn concat(&self, py: Python<'_>, other: ListAdd<'_, Self>) -> PyResult<Self> {
        let out = match other {
            ListAdd::Identity => self.values.repeat(py, 2),
            ListAdd::Sorted(list) => self
                .values
                .iter()
                .chain(list.values.iter())
                .map(|x| x.clone_ref(py))
                .collect(),
            ListAdd::Vec(values) => values,
            ListAdd::Iterator(it) => self
                .values
                .iter()
                .map(|x| x.clone_ref(py).pipe(Ok::<Py<PyAny>, PyErr>))
                .chain(it.map_ok(Bound::unbind))
                .collect::<PyResult<_>>()?,
        };
        self.as_owned_from(py, out)
    }
    fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new(self, pos) {
            ops::Expand::MustSplit => {
                let (half, last_max) = self.expand_splitted(py, pos);
                self.maxes[pos] = self.keys()[pos].last().unwrap().clone_ref(py);
                self.maxes.insert(pos + 1, last_max);
                self.values.insert(pos + 1, half);
                self.lengths.inner.clear();
            }
            ops::Expand::IdxBuilt => self.lengths.expand_at(pos),
            ops::Expand::NoOp => (),
        }
    }
    fn get_item_or_slice<'py>(&mut self, index: IntOrSlice<'py>) -> PyResult<ListOrAny<'py>> {
        match index {
            Either::Right(slice) => SliceIdxs::new(&slice, self.len())?
                .extract_from(slice.py(), self)?
                .iter()
                .collect_bound(slice.py())
                .map(Either::Left),
            Either::Left(index) => self
                .get_item(index.py(), index.extract()?)
                .map(Either::Right),
        }
    }
    fn del_item_or_slice(&mut self, index: IntOrSlice<'_>) -> PyResult<()> {
        match index {
            Either::Right(slice) => {
                SliceIdxs::new(&slice, self.len())?.delete_from(slice.py(), self)
            }
            Either::Left(index) => self.del_item(index.py(), index.extract()?),
        }
    }
    fn del_item(&mut self, py: Python<'_>, index: isize) -> PyResult<()> {
        if self.is_empty() {
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
        let values = self.values.repeat(py, num);
        self.clear(py);
        self.extend(py, values.into())
    }
    fn index(&mut self, value: &Bound<'_, PyAny>, (start, stop): RawRangeIdxs) -> PyResult<usize> {
        Bounds::new(start, stop)
            .try_normalize(self)
            .ok_or_else(|| errors::not_in_list(value))
            .and_then(|indices| self.index_inner(value, indices))
    }
    fn pop<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let loc = self.index_to_loc(index)?;
        let val = self.values.loc(&loc).clone_ref(py);
        self.delete(py, loc)?;
        Ok(val.into_bound(py))
    }
    #[inline(always)]
    fn push_chunk(&mut self, py: Python<'_>, chunk: VecPy) {
        self.maxes.push(chunk.last().unwrap().clone_ref(py));
        self.keys_mut().push(chunk);
    }
    #[inline(always)]
    fn push_val(&mut self, py: Python<'_>, pos: usize, value: Py<PyAny>) {
        self.maxes[pos] = value.clone_ref(py);
        self.keys_mut()[pos].push(value);
    }
    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.find(value)? {
            Some(loc) => self.delete(value.py(), loc),
            None => Err(errors::not_in_list(value)),
        }
    }
    fn as_repeated(&self, py: Python<'_>, num: usize) -> PyResult<Self> {
        self.as_owned_from(py, self.values.repeat(py, num))
    }
    fn reset(&mut self, py: Python<'_>, load: usize) -> PyResult<()> {
        let values = self.values.iter_owned(py).collect_vec();
        self.clear(py);
        self.load = load.into();
        self.extend(py, values.into())
    }
    fn set_item(&mut self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.del_item(value.py(), index)?;
        self.append(value)
    }

    #[inline(always)]
    fn try_bisect<const LEFT: bool>(&self, value: &Bound<'_, PyAny>) -> BisectRes {
        ops::Bisect::new::<LEFT, Self>(self, value)
    }
    #[inline(always)]
    fn try_bisect_left(&self, value: &Bound<'_, PyAny>) -> BisectRes {
        self.try_bisect::<true>(value)
    }
    #[inline(always)]
    fn try_bisect_right(&self, value: &Bound<'_, PyAny>) -> BisectRes {
        self.try_bisect::<false>(value)
    }
    fn bisect_by_key<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.try_bisect::<LEFT>(value)? {
            ops::Bisect::Empty => Ok(0),
            ops::Bisect::LenEQPos => Ok(self.len()),
            ops::Bisect::LenNEPos(loc) => Ok(loc.into_index(self)),
        }
    }
    fn delete_many(&mut self, loc: &mut Loc) -> usize {
        if loc.pos == 0 {
            loc.pos += 1;
        }
        let prev = loc.pos - 1;
        let mut removed = self.remove_pos(loc);
        self.values[prev].append(removed.as_mut());
        prev
    }
}
