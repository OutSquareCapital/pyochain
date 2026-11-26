use std::ops::{Deref, DerefMut};

use crate::{
    bounds::{Bounds, Loc, RawIrangeSpecs},
    chunks::ChunksMethods,
    errors,
    get_item::Index,
    inner::InnerList,
    iter::{IrangeBounds, builder::RawRangeIdxs},
    list_update::IntoList,
    ops,
    repeat::Repeated,
    reprs::PyRepr,
    sort::KeyFn,
    types::{IndexOut, IntoIndex},
};
use itertools::Itertools;
use pyo3::prelude::*;
use tap::prelude::*;

pub type BisectRes = PyResult<ops::Bisect>;
pub type ItemOf<L> = <<L as ListsMethods>::Chunks as ChunksMethods>::Item;
pub trait ListsMethods:
    Deref<Target = InnerList<Self::Chunks>> + DerefMut + PyRepr + Sized + Send + Sync
{
    type Chunks: ChunksMethods;
    type Key<'py>: KeyFn<'py, Item = ItemOf<Self>>;
    fn key_func<'py>(&self, py: Python<'py>) -> Self::Key<'py>;
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    #[must_use]
    fn new_empty(&self, py: Python<'_>) -> Self;
    #[must_use]
    fn copy(&self, py: Python<'_>) -> Self;
    /// Equivalent to `bisect_ky_key` if the collection is sorted **by value**.\
    /// If the collection is sorted **by key**, *value* is first transformed by the *key* function before the bisect operation.
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>>;
    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize>;
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
    fn concat(&self, py: Python<'_>, other: IntoList) -> PyResult<Self> {
        other.concat_into(py, self)
    }
    fn extend(&mut self, py: Python<'_>, values: IntoList) -> PyResult<()> {
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
    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        match self.find(&value) {
            Ok(Some(loc)) => {
                self.delete(value.py(), loc);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(e) => Err(e),
        }
    }
    fn imul(&mut self, py: Python<'_>, num: usize) -> PyResult<()> {
        *self = Repeated::new(num).extend_into(py, self)?;
        Ok(())
    }
    fn mul(&self, py: Python<'_>, num: usize) -> PyResult<Self> {
        Repeated::new(num).extend_into(py, self)
    }
    fn index(&mut self, value: &Bound<'_, PyAny>, (start, stop): RawRangeIdxs) -> PyResult<usize> {
        Bounds::new(start, stop)
            .try_normalize(self)
            .ok_or_else(|| errors::not_in_list(value))
            .and_then(|indices| self.index_inner(value, indices))
    }
    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.find(value)?
            .map(|loc| self.delete(value.py(), loc))
            .ok_or_else(|| errors::not_in_list(value))
    }
    fn reset(&mut self, load: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = load.py();
        let values = self.chunks.values().iter_owned(py).collect_vec();
        self.load = load.try_into()?;
        self.clear();
        values.conv::<IntoList>().extend_into(py, self)
    }
    fn set_item(&mut self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.del_item(value.py(), index)?;
        self.append(value)
    }
}
