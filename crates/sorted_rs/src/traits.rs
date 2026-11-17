use std::ops::{Deref, DerefMut};

use crate::{
    bounds::{Bounds, Loc, RawIrangeSpecs},
    errors,
    get_item::Index,
    inner::{InnerList, Values},
    iter::{IrangeBounds, builder::RawRangeIdxs},
    list_update::{KeyFn, ListAdd},
    ops,
    reprs::PyRepr,
    types::{IndexOut, IntoIndex, VecPy},
};
use itertools::Itertools;
use pyo3::prelude::*;
use tap::prelude::*;

pub type BisectRes = PyResult<ops::Bisect>;
pub trait ListsMethods:
    Deref<Target = InnerList> + DerefMut + PyRepr + Sized + Send + Sync
{
    /// Returns a reference to the "keys" of the collection.\
    /// For collections sorted by *values*, this is equivalent to `InnerData::values`.\
    /// For collections sorted by *key*, this is equivalent to `KeysListsData::1`.
    const IS_KEYED: bool;
    fn keys(&self) -> &Values;
    fn keys_mut(&mut self) -> &mut Values;
    fn key_func<'py>(&self, py: Python<'py>) -> impl KeyFn<'py> + use<'py, Self>;
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    #[must_use]
    fn new_empty(&self, py: Python<'_>) -> Self;
    #[must_use]
    fn copy(&self, py: Python<'_>) -> Self;
    /// Equivalent to `bisect_ky_key` if the collection is sorted **by value**.\
    /// If the collection is sorted **by key**, *value* is first transformed by the *key* function before the bisect operation.
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn clear(&mut self, py: Python<'_>);
    fn delete(&mut self, py: Python<'_>, loc: Loc) -> PyResult<()>;
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>>;
    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize>;
    fn split_values(&mut self, pos: usize, load: usize);
    fn irange<'py>(
        &self,
        py: Python<'py>,
        specs: RawIrangeSpecs<'py>,
    ) -> PyResult<IrangeBounds<'py>>;
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    #[inline]
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.find(value).map(|x| x.is_some())
    }
    fn concat(&self, py: Python<'_>, other: ListAdd<'_>) -> PyResult<Self> {
        other.concat_into(py, self)
    }
    fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new(self, pos) {
            ops::Expand::MustSplit => {
                let load = *self.load;
                let half = self.keys_mut()[pos].split_off(load);
                let last_max = half.last().unwrap().clone_ref(py);
                self.maxes.insert(pos + 1, last_max);
                self.maxes[pos] = self.keys()[pos].last().unwrap().clone_ref(py);
                self.keys_mut().insert(pos + 1, half);
                self.split_values(pos, load);
                self.lengths.inner.clear();
            }
            ops::Expand::IdxBuilt => self.lengths.expand_at(pos),
            ops::Expand::NoOp => (),
        }
    }
    fn extend(&mut self, py: Python<'_>, values: ListAdd<'_>) -> PyResult<()> {
        values.extend_into(py, self)
    }
    fn get_item_or_slice<'py>(
        &mut self,
        py: Python<'py>,
        index: IntoIndex<'py>,
    ) -> PyResult<IndexOut<'py>> {
        Index::new(py, index).get_from_list(self)
    }
    fn del_item_or_slice(&mut self, py: Python<'_>, index: IntoIndex<'_>) -> PyResult<()> {
        Index::new(py, index).del_from_list(self)
    }
    fn del_item(&mut self, py: Python<'_>, index: isize) -> PyResult<()> {
        if self.is_empty() {
            Err(errors::out_of_range())
        } else {
            let loc = self.set_pos(index)?;
            self.delete(py, loc)
        }
    }
    fn delete_on_len_sup_to_half_load(&mut self, py: Python<'_>, loc: Loc) {
        self.maxes[loc.pos] = self.keys().loc_last(&loc).clone_ref(py);
        if self.lengths.is_built() {
            self.lengths.delete_at(loc.pos);
        }
    }
    fn delete_many(&mut self, loc: &mut Loc) -> usize {
        if loc.pos == 0 {
            loc.pos += 1;
        }
        let prev = loc.pos - 1;
        let mut removed = self.remove_at(loc);
        self.values[prev].append(removed.as_mut());
        prev
    }
    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.find(&value)?
            .map_or(Ok(()), |loc| self.delete(value.py(), loc))
    }
    fn imul(&mut self, py: Python<'_>, num: usize) -> PyResult<()> {
        let values = self.values.repeat(py, num)?;
        self.clear(py);
        self.extend(py, values)
    }
    fn mul(&self, py: Python<'_>, num: usize) -> PyResult<Self> {
        self.values.repeat(py, num)?.into_new_list(py, self)
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
    fn push_val(&mut self, py: Python<'_>, value: Py<PyAny>) {
        let pos = self.maxes.len() - 1;
        self.maxes[pos] = value.clone_ref(py);
        self.keys_mut()[pos].push(value);
        self.expand(py, pos);
    }
    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        match self.find(value)? {
            Some(loc) => self.delete(value.py(), loc),
            None => Err(errors::not_in_list(value)),
        }
    }
    fn reset(&mut self, py: Python<'_>, load: usize) -> PyResult<()> {
        let values = self.values.iter_owned(py).collect_vec();
        self.clear(py);
        self.load = load.into();
        values
            .into_iter()
            .pipe(ListAdd::Sorted)
            .extend_into(py, self)
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
            ops::Bisect::MaxesLen => Ok(self.len()),
            ops::Bisect::AtLoc(loc) => Ok(loc.into_index(self)),
        }
    }
}
