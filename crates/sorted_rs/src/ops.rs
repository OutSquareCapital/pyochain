//! Logical kinds of operations for sorted lists.\
//! Those enums allow to first make cheap conditional checks, and then match them on clearly named branches in a flat match statement, instead of having nested if statements with complex conditions.\
//! More importantly, they factorize the common logic between the `SortedList` and `SortedKeyList` implementations.\
//! They have very similar conditional checks, but different actions to take on each branch.

use std::{cmp::Ordering, hint::unreachable_unchecked};

use pyo3::prelude::*;
use tap::Conv;

use crate::{
    bisect::Bisect, bounds::Loc, errors, inner::InnerData, lengths::Lengths, traits::NestedVec,
};

pub(super) enum Nb {
    GTOne,
    One,
    Zero,
    MinusOne,
    LTMinusOne,
}
impl From<isize> for Nb {
    fn from(value: isize) -> Self {
        match value {
            0 => Self::Zero,
            1 => Self::One,
            -1 => Self::MinusOne,
            _ if value > 1 => Self::GTOne,
            _ => Self::LTMinusOne,
        }
    }
}
pub(super) enum Slice {
    /// Whole slice optimization: start to stop slices the whole sorted list.
    Full,
    Forward,
    ForwardToEnd,
    ForwardSmall,
    ForwardSmallToAdjust,
    ForwardStep,
    Backward,
    BackwardStep,
    ///A negative step could reverse the order of the items and this could be the desired behavior.
    MayReverse,
    /// Negative step with nothing to iterate (mirrors Python's `range`, which is empty when `start <= stop` for a negative step).
    NoOp,
}
impl Slice {
    pub fn new(start: isize, stop: isize, step: isize, length: isize) -> Self {
        debug_assert!(stop <= length);
        debug_assert_ne!(step, 0);
        match (start.cmp(&stop), step.conv::<Nb>()) {
            // SAFETY: Constructors guarantee that `step` is not zero (slice.indices raises `ValueError` on `step == 0`).
            (_, Nb::Zero) => unsafe { unreachable_unchecked() },
            (Ordering::Less, Nb::One) => match (stop == length, length <= 8 * (stop - start)) {
                (true, _) if start == 0 => Self::Full,
                (true, true) => Self::ForwardSmall,
                (true, false) => Self::ForwardToEnd,
                (false, true) => Self::ForwardSmallToAdjust,
                (false, false) => Self::Forward,
            },
            (Ordering::Less, Nb::GTOne) => Self::ForwardStep,
            (Ordering::Equal | Ordering::Greater, Nb::One | Nb::GTOne) => Self::MayReverse,
            (Ordering::Greater, Nb::MinusOne) => Self::Backward,
            (Ordering::Greater, Nb::LTMinusOne) => Self::BackwardStep,
            (Ordering::Less | Ordering::Equal, Nb::MinusOne | Nb::LTMinusOne) => Self::NoOp,
        }
    }
}
/// Used in `add`, `discard`, `__contains__`, `count`, and `remove`
pub(super) enum Maxes {
    Empty,
    LenEQPos(Loc),
    LenNEPos(Loc),
    BisectErr(PyErr),
}
impl Maxes {
    #[inline(always)]
    pub fn left(maxes: &[Py<PyAny>], value: &Bound<'_, PyAny>) -> Self {
        Self::new::<true>(maxes, value)
    }
    #[inline(always)]
    pub fn right(maxes: &[Py<PyAny>], value: &Bound<'_, PyAny>) -> Self {
        Self::new::<false>(maxes, value)
    }
    #[inline(always)]
    fn new<const LEFT: bool>(maxes: &[Py<PyAny>], value: &Bound<'_, PyAny>) -> Self {
        if maxes.is_empty() {
            Self::Empty
        } else {
            if LEFT {
                maxes.bisect_left(value)
            } else {
                maxes.bisect_right(value)
            }
            .map(Loc::with_pos)
            .map_or_else(Self::BisectErr, |bound| {
                if bound.pos == maxes.len() {
                    Self::LenEQPos(bound)
                } else {
                    Self::LenNEPos(bound)
                }
            })
        }
    }
}

/// Used in `expand`
pub enum Expand {
    PosLenGtLoad,
    IdxBuilt,
    Other,
}
impl Expand {
    #[inline(always)]
    #[must_use]
    pub fn new(lists_len: usize, load: usize, lengths: &Lengths) -> Self {
        if lists_len > load << 1 {
            Self::PosLenGtLoad
        } else if lengths.is_built() {
            Self::IdxBuilt
        } else {
            Self::Other
        }
    }
}
/// Used in `delete`
pub enum Delete {
    PosSupToLoad,
    DataLenGTOne,
    LenPosNotZero,
    Other,
}
impl Delete {
    #[inline(always)]
    #[must_use]
    pub fn new(lists: &[Vec<Py<PyAny>>], load: usize, loc: &Loc) -> Self {
        let len_pos = lists.loc_len(loc);
        if len_pos > (load >> 1) {
            Self::PosSupToLoad
        } else if lists.len() > 1 {
            Self::DataLenGTOne
        } else if len_pos != 0 {
            Self::LenPosNotZero
        } else {
            Self::Other
        }
    }
}
/// `Loc`, `start`, and `stop` bounds for a search in a sorted list.
type IdxBounds = (Loc, usize, usize);
pub(super) enum Index<'py, 'a> {
    NotFound(&'a Bound<'py, PyAny>),
    Empty(&'a Bound<'py, PyAny>),
    InvalidRange(&'a Bound<'py, PyAny>),
    BisectErr(PyErr),
    Searchable(IdxBounds),
}
impl<'py, 'a> Index<'py, 'a> {
    pub fn new(
        data: &InnerData,
        value: &'a Bound<'py, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> Self {
        let length = data.len.cast_signed();
        if length == 0 {
            Self::Empty(value)
        } else {
            let mut start = start.unwrap_or(0);
            let mut stop = stop.unwrap_or(length);
            if start < 0 {
                start += length;
            }
            start = start.max(0);
            if stop < 0 {
                stop += length;
            }
            stop = stop.min(length);
            if stop <= start {
                Self::InvalidRange(value)
            } else {
                match data.maxes.bisect_left(value).map(Loc::with_pos) {
                    Ok(bound) if bound.pos == data.maxes.len() => Self::NotFound(value),
                    Ok(bound) => {
                        Self::Searchable((bound, start.cast_unsigned(), stop.cast_unsigned()))
                    }
                    Err(err) => Self::BisectErr(err),
                }
            }
        }
    }
    pub fn into_res(self) -> PyResult<IdxBounds> {
        match self {
            Self::Searchable((bound, start, stop)) => Ok((bound, start, stop)),
            Self::NotFound(value) | Self::Empty(value) | Self::InvalidRange(value) => {
                Err(errors::not_in_list(&value.repr()?))
            }
            Self::BisectErr(err) => Err(err),
        }
    }
}
