use crate::{
    ListGetter,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    chunks::ChunksMethods,
    inner::InnerList,
    iter::{BoundedFwd, IterMethods, PySortedIter, core::InnerRef, cursor::Cursor},
    ops,
    traits::ListsMethods,
    types::Pair,
};
use pyo3::prelude::*;
use tap::prelude::*;

pub type RawRangeIdxs = Pair<Option<isize>>;
type IterBounds = Option<PyResult<Pair<Loc>>>;

#[derive(Debug)]
pub struct Builder<'a, 'py, T: ListGetter> {
    pyref: &'a Bound<'py, T>,
    inner: &'a T,
    bounds: IterBounds,
}
impl<'a, 'py, T: ListGetter> Builder<'a, 'py, T>
where
    T: ListGetter,
{
    #[must_use]
    pub fn new(pyref: &'a Bound<'py, T>) -> Self {
        Self {
            pyref,
            inner: pyref.get(),
            bounds: None,
        }
    }
    pub fn build<I>(self) -> PyResult<Bound<'py, I>>
    where
        I: PySortedIter<Inner = BoundedFwd>,
    {
        let (min, max) = self.bounds.transpose()?.unwrap_or_default();
        InnerRef::from_owner(self.pyref, self.inner)
            .into_iterator(min.conv::<Cursor>().into_bounded(max.into()))
            .into_py(self.pyref.py())
    }
    #[must_use]
    pub fn islice(mut self, specs: RawRangeIdxs) -> Self {
        self.bounds = self
            .inner
            .write()
            .as_mut()
            .islice_bounds(specs.conv::<Bounds<_>>());
        self
    }
    pub fn irange<const BY_KEY: bool>(mut self, specs: RawIrangeSpecs) -> PyResult<Self> {
        let guard = self.inner.read();
        let bounds = if BY_KEY {
            specs.conv::<IrangeBounds>()
        } else {
            guard.as_ref().irange(specs)?
        };
        self.bounds = guard.as_ref().irange_bounds(bounds);
        Ok(self)
    }
}
impl<C: ChunksMethods> InnerList<C> {
    fn irange_bounds(&self, specs: IrangeBounds<'_>) -> IterBounds {
        if self.is_empty() {
            None
        } else {
            let min = match specs.start {
                None => Ok(Loc::default()),
                Some(BoundKind::Inclusive(m)) => self.get_bound::<true, false>(&m)?,
                Some(BoundKind::Exclusive(m)) => self.get_bound::<false, false>(&m)?,
            };
            let max = match specs.stop {
                None => Ok(self.last_sublist()),
                Some(BoundKind::Inclusive(m)) => self.get_bound::<false, true>(&m)?,
                Some(BoundKind::Exclusive(m)) => self.get_bound::<true, true>(&m)?,
            };
            match (min, max) {
                (_, Err(e)) | (Err(e), _) => Some(Err(e)),
                (Ok(min), Ok(max))
                    if min.pos > max.pos || (min.pos == max.pos && min.idx >= max.idx) =>
                {
                    None
                }
                (Ok(min), Ok(max)) => Some(Ok((min, max))),
            }
        }
    }
    fn islice_bounds(&mut self, specs: Bounds<Option<isize>>) -> IterBounds {
        let bounds = specs.try_normalize(self)?;
        let min = self.set_pos_usize(bounds.start);
        let max = if bounds.stop == self.len() {
            let pos = self.chunks.values().len() - 1;
            let idx = self.chunks.values()[pos].len();
            Ok(Loc::new(pos, idx))
        } else {
            self.set_pos_usize(bounds.stop)
        };
        match (min, max) {
            (Err(e), _) | (_, Err(e)) => Some(Err(e)),
            (Ok(min), Ok(max)) => Some(Ok((min, max))),
        }
    }
    fn get_bound<const LEFT: bool, const MAX: bool>(
        &self,
        value: &Bound<'_, PyAny>,
    ) -> Option<PyResult<Loc>> {
        match self.try_bisect::<LEFT>(value) {
            Err(e) => Some(Err(e)),
            Ok(ops::Bisect::MaxesLen(_)) if MAX => Some(Ok(self.last_sublist())),
            Ok(ops::Bisect::Empty | ops::Bisect::MaxesLen(_)) => None,
            Ok(ops::Bisect::AtLoc(loc)) => Some(Ok(loc)),
        }
    }
    #[inline]
    fn last_sublist(&self) -> Loc {
        let pos = self.maxes.len() - 1;
        let idx = self.chunks.keys()[pos].len();
        Loc::new(pos, idx)
    }
}
pub type IrangeBounds<'py> = Bounds<Option<BoundKind<'py>>>;
impl<'py> From<RawIrangeSpecs<'py>> for IrangeBounds<'py> {
    fn from((min, max, (incl_min, incl_max)): RawIrangeSpecs<'py>) -> Self {
        Self::new(BoundKind::new(min, incl_min), BoundKind::new(max, incl_max))
    }
}
impl<'py> IrangeBounds<'py> {
    pub fn with_keys(values: impl Into<Self>, func: &Bound<'py, PyAny>) -> PyResult<Self> {
        let mut slf = values.into();
        slf.start = slf.start.map(|x| x.with_key(func)).transpose()?;
        slf.stop = slf.stop.map(|x| x.with_key(func)).transpose()?;
        Ok(slf)
    }
}

#[derive(Debug, Clone)]
pub enum BoundKind<'py> {
    Inclusive(Bound<'py, PyAny>),
    Exclusive(Bound<'py, PyAny>),
}
impl<'py> BoundKind<'py> {
    #[inline]
    fn new(value: Option<Bound<'py, PyAny>>, inclusive: bool) -> Option<Self> {
        match (value, inclusive) {
            (None, _) => None,
            (Some(v), true) => Some(Self::Inclusive(v)),
            (Some(v), false) => Some(Self::Exclusive(v)),
        }
    }
    #[inline]
    fn with_key<'a>(self, key: &'a Bound<'py, PyAny>) -> PyResult<Self> {
        match self {
            Self::Inclusive(value) => key.call1((value,)).map(Self::Inclusive),
            Self::Exclusive(value) => key.call1((value,)).map(Self::Exclusive),
        }
    }
}
