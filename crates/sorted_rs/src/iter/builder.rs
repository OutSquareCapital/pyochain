use crate::{
    ListGetter,
    bisect::Bisect,
    bounds::{Loc, RangeIdxs, RawIrangeSpecs},
    inner::InnerList,
    iter::{BoundedFwd, IterMethods, PySortedIter, core::InnerRef, cursor::Cursor},
    traits::ListsMethods,
};
use derive_more::Constructor;
use pyo3::prelude::*;
use tap::{Conv, Pipe};

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
        self.bounds = islice_bounds(self.inner.write().as_mut(), specs.conv::<RangeIdxs<_>>());
        self
    }
    pub fn irange<const BY_KEY: bool>(mut self, specs: RawIrangeSpecs) -> PyResult<Self> {
        let guard = self.inner.read();
        let bounds = if BY_KEY {
            specs.conv::<IrangeSpecs>()
        } else {
            guard.as_ref().irange(self.pyref.py(), specs)?
        };
        self.bounds = irange_bounds(guard.as_ref(), bounds);
        Ok(self)
    }
}

fn irange_bounds<T>(data: &T, specs: IrangeSpecs<'_>) -> IterBounds
where
    T: ListsMethods,
{
    if data.maxes.is_empty() {
        None
    } else {
        let min = match specs.min {
            RangeBound::None => Ok(Loc::default()),
            RangeBound::Inclusive(m) => min_bound::<true, T>(data, &m)?,
            RangeBound::Exclusive(m) => min_bound::<false, T>(data, &m)?,
        };
        let max = match specs.max {
            RangeBound::None => Ok(new_inclusive_max(data)),
            RangeBound::Inclusive(m) => max_bound::<false, T>(data, &m),
            RangeBound::Exclusive(m) => max_bound::<true, T>(data, &m),
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
fn islice_bounds(list: &mut InnerList, specs: RangeIdxs<Option<isize>>) -> IterBounds {
    if list.is_empty() {
        None
    } else {
        let length = list.len();
        let normalize_bound = |mut val: isize| {
            if val.is_negative() {
                val += length.cast_signed();
                val.max(0).cast_unsigned()
            } else {
                val.cast_unsigned().min(length)
            }
        };
        let start = specs.start.map_or_default(normalize_bound);
        let stop = specs.stop.map_or(length, normalize_bound);
        if start >= stop {
            None
        } else {
            let min = list.set_pos_usize(start);
            let max = if stop == length {
                let pos = list.values.len() - 1;
                let idx = list.values.last().unwrap().len();
                Ok(Loc::new(pos, idx))
            } else {
                list.set_pos_usize(stop)
            };
            match (min, max) {
                (Err(e), _) | (_, Err(e)) => Some(Err(e)),
                (Ok(min), Ok(max)) => Some(Ok((min, max))),
            }
        }
    }
}
fn min_bound<const LEFT: bool, T>(data: &T, value: &Bound<'_, PyAny>) -> Option<PyResult<Loc>>
where
    T: ListsMethods,
{
    match data.maxes.bisect::<LEFT>(value) {
        Err(e) => Some(Err(e)),
        Ok(pos) if pos == data.maxes.len() => None,
        Ok(pos) => data.keys()[pos]
            .bisect::<LEFT>(value)
            .map(|idx| Loc::new(pos, idx))
            .pipe(Some),
    }
}
fn max_bound<const LEFT: bool, T>(data: &T, value: &Bound<'_, PyAny>) -> PyResult<Loc>
where
    T: ListsMethods,
{
    let mut pos = data.maxes.bisect::<LEFT>(value)?;
    let idx = if pos == data.maxes.len() {
        pos -= 1;
        data.keys()[pos].len()
    } else {
        data.keys()[pos].bisect::<LEFT>(value)?
    };
    Ok(Loc::new(pos, idx))
}
#[derive(Constructor)]
pub struct IrangeSpecs<'py> {
    min: RangeBound<'py>,
    max: RangeBound<'py>,
}
impl<'py> From<RawIrangeSpecs<'py>> for IrangeSpecs<'py> {
    fn from((min, max, (incl_min, incl_max)): RawIrangeSpecs<'py>) -> Self {
        Self {
            min: RangeBound::new(min, incl_min),
            max: RangeBound::new(max, incl_max),
        }
    }
}
impl<'py> IrangeSpecs<'py> {
    pub fn with_keys(py: Python<'py>, values: impl Into<Self>, key: &Py<PyAny>) -> PyResult<Self> {
        let mut slf = values.into();
        let func = key.bind(py);
        slf.min = slf.min.with_key(func)?;
        slf.max = slf.max.with_key(func)?;
        Ok(slf)
    }
}
#[inline]
fn new_inclusive_max<T: ListsMethods>(data: &T) -> Loc {
    let pos = data.maxes.len() - 1;
    let idx = data.keys()[pos].len();
    Loc::new(pos, idx)
}

#[derive(Debug, Clone)]
enum RangeBound<'py> {
    None,
    Inclusive(Bound<'py, PyAny>),
    Exclusive(Bound<'py, PyAny>),
}
impl<'py> RangeBound<'py> {
    #[inline]
    fn new(value: Option<Bound<'py, PyAny>>, inclusive: bool) -> Self {
        match (value, inclusive) {
            (None, _) => Self::None,
            (Some(v), true) => Self::Inclusive(v),
            (Some(v), false) => Self::Exclusive(v),
        }
    }
    #[inline]
    fn with_key<'a>(self, key: &'a Bound<'py, PyAny>) -> PyResult<Self> {
        match self {
            Self::None => Ok(Self::None),
            Self::Inclusive(value) => key.call1((value,)).map(Self::Inclusive),
            Self::Exclusive(value) => key.call1((value,)).map(Self::Exclusive),
        }
    }
}
