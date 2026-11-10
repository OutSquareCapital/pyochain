use crate::{
    ListGetter,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    inner::InnerList,
    iter::{BoundedFwd, IterMethods, PySortedIter, core::InnerRef, cursor::Cursor},
    ops,
    traits::ListsMethods,
};
use pyo3::prelude::*;
use tap::prelude::*;

pub type RawRangeIdxs = (Option<isize>, Option<isize>);
type IterBounds = Option<PyResult<(Loc, Loc)>>;

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
        self.bounds = islice_bounds(self.inner.write().as_mut(), specs.conv::<Bounds<_>>());
        self
    }
    pub fn irange<const BY_KEY: bool>(mut self, specs: RawIrangeSpecs) -> PyResult<Self> {
        let guard = self.inner.read();
        let bounds = if BY_KEY {
            specs.conv::<IrangeBounds>()
        } else {
            guard.as_ref().irange(self.pyref.py(), specs)?
        };
        self.bounds = irange_bounds(guard.as_ref(), bounds);
        Ok(self)
    }
}

fn irange_bounds<T>(list: &T, specs: IrangeBounds<'_>) -> IterBounds
where
    T: ListsMethods,
{
    if list.maxes.is_empty() {
        None
    } else {
        let min = match specs.start {
            None => Ok(Loc::default()),
            Some(BoundKind::Inclusive(m)) => get_bound::<true, false, T>(list, &m)?,
            Some(BoundKind::Exclusive(m)) => get_bound::<false, false, T>(list, &m)?,
        };
        let max = match specs.stop {
            None => Ok(last_sublist(list)),
            Some(BoundKind::Inclusive(m)) => get_bound::<false, true, T>(list, &m)?,
            Some(BoundKind::Exclusive(m)) => get_bound::<true, true, T>(list, &m)?,
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
fn islice_bounds(list: &mut InnerList, specs: Bounds<Option<isize>>) -> IterBounds {
    let bounds = specs.try_normalize(list)?;
    let min = list.set_pos_usize(bounds.start);
    let max = if bounds.stop == list.len() {
        let pos = list.values.len() - 1;
        let idx = list.values[pos].len();
        Ok(Loc::new(pos, idx))
    } else {
        list.set_pos_usize(bounds.stop)
    };
    match (min, max) {
        (Err(e), _) | (_, Err(e)) => Some(Err(e)),
        (Ok(min), Ok(max)) => Some(Ok((min, max))),
    }
}
fn get_bound<const LEFT: bool, const MAX: bool, T>(
    list: &T,
    value: &Bound<'_, PyAny>,
) -> Option<PyResult<Loc>>
where
    T: ListsMethods,
{
    match list.try_bisect::<LEFT>(value) {
        Err(e) => Some(Err(e)),
        Ok(ops::Bisect::LenEQPos) if MAX => Some(Ok(last_sublist(list))),
        Ok(ops::Bisect::Empty | ops::Bisect::LenEQPos) => None,
        Ok(ops::Bisect::LenNEPos(loc)) => Some(Ok(loc)),
    }
}
#[inline]
fn last_sublist<T: ListsMethods>(list: &T) -> Loc {
    let pos = list.maxes.len() - 1;
    let idx = list.keys()[pos].len();
    Loc::new(pos, idx)
}
type IrangeBounds<'py> = Bounds<Option<BoundKind<'py>>>;
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
enum BoundKind<'py> {
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
