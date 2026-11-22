use std::cmp::Ordering;

use derive_more::From;
use pyo3::{
    ffi,
    prelude::*,
    types::{PyFloat, PyInt, PyString, PyTuple},
};
use pyo3_ext::{prelude::*, pystring::UnicodeByteKind};
use pyochain_macros::try_cast;
use tap::prelude::*;

use crate::sort::SortKey;

/// The actual, high level dispatch type for the sort strategy.\
/// The two variants simply answer: "Is the comparison done on a tuple or a single element?"
#[derive(Debug, Clone, Copy)]
pub(super) enum SortBy {
    Tuple(Element),
    Element(Element),
}
impl SortBy {
    pub fn check_first(first: &impl SortKey) -> Self {
        let first = first.key();
        match try_get_first(first) {
            Some(key) if key.is_exact_instance_of::<PyTuple>() => Self::Tuple(Object::Safe.into()),
            Some(key) => Self::Tuple(key.into()),
            None => Self::Element(first.into()),
        }
    }
    pub fn check_next(self, item: &Bound<'_, PyAny>) -> Option<Self> {
        match self {
            // Nothing left to check: stop.
            Self::Element(Element::Object(Object::Safe)) => None,
            // See `Element::check`.
            Self::Element(s) => s.check_next(item).map(SortBy::Element),
            // Stops if the item isn't a non-empty tuple, otherwise `Element::check` on its first item.
            // Like CPython, a mismatch on the first items keeps the tuple compare, with the safe compare on them.
            Self::Tuple(s) => try_get_first(item)
                .map(|key| s.check_next(key).unwrap_or(Element::SAFE))
                .map(SortBy::Tuple),
        }
    }
    pub(super) fn sort<T: SortKey>(self, mut items: Vec<T>) -> PyResult<Vec<T>> {
        use Builtin as Bl;
        use Element as El;
        use Object as Ob;
        use SortBy as Sb;
        let mut try_cmp = TryCmp::default();
        match self {
            Sb::Element(El::Object(Ob::Safe)) => {
                items.sort_by(|a, b| try_cmp.catch(|| a.object(b)));
            }
            Sb::Element(El::Builtin(Bl::Int)) => items.sort_by(SortKey::long),
            Sb::Element(El::Builtin(Bl::Float)) => items.sort_by(SortKey::float),
            Sb::Element(El::Builtin(Bl::String)) => items.sort_by(SortKey::latin),
            Sb::Element(El::Object(Ob::Unsafe { func, .. })) => {
                items.sort_by(|a, b| try_cmp.catch(|| a.unsafe_object(b, func)));
            }
            Sb::Tuple(El::Builtin(Bl::Int)) => {
                items.sort_by(|a, b| try_cmp.catch(|| a.tuple(b, SortKey::long)));
            }
            Sb::Tuple(El::Builtin(Bl::Float)) => {
                items.sort_by(|a, b| try_cmp.catch(|| a.tuple(b, SortKey::float)));
            }
            Sb::Tuple(El::Builtin(Bl::String)) => {
                items.sort_by(|a, b| try_cmp.catch(|| a.tuple(b, SortKey::latin)));
            }
            Sb::Tuple(El::Object(Ob::Safe)) => {
                items.sort_by(|a, b| try_cmp.catch(|| a.try_tuple(b, SortKey::object)));
            }
            Sb::Tuple(El::Object(Ob::Unsafe { func, .. })) => {
                items.sort_by(|a, b| {
                    try_cmp.catch(|| a.try_tuple(b, |a, b| SortKey::unsafe_object(a, b, func)))
                });
            }
        }
        try_cmp.0.map_or(Ok(items), Err)
    }
}
/// First item of a non-empty exact tuple.
fn try_get_first<'a, 'py>(item: &'a Bound<'py, PyAny>) -> Option<&'a Bound<'py, PyAny>> {
    item.cast_exact::<PyTuple>().ok()?.as_slice().first()
}
/// First error raised by a comparison during the sort.\
/// Once set, the remaining comparisons don't call Python anymore: the order is discarded anyway.
#[derive(Default)]
struct TryCmp(Option<PyErr>);
impl TryCmp {
    #[inline(always)]
    fn catch(&mut self, cmp: impl FnOnce() -> PyResult<Ordering>) -> Ordering {
        if self.0.is_some() {
            Ordering::Equal
        } else {
            cmp().unwrap_or_else(|e| {
                self.0 = Some(e);
                Ordering::Equal
            })
        }
    }
}
/// Is the element comparison a builtin with a specialized fast path, or a generic object comparison?
#[derive(Debug, Clone, Copy, From)]
pub(super) enum Element {
    Builtin(Builtin),
    Object(Object),
}
impl Default for Element {
    fn default() -> Self {
        Self::SAFE
    }
}
impl Element {
    pub const SAFE: Self = Self::Object(Object::Safe);
    fn from_object<T>(obj: &Bound<'_, T>) -> Self {
        obj.as_any().get_type_ptr().conv::<Object>().into()
    }
    /// Keeps `self` if `key` has the same type and still fits the fast path.\
    /// A non-compact int or a non-latin string falls back to the homogeneous compare of its type,
    /// any other type mismatch to the safe compare.
    pub fn check_next(self, key: &Bound<'_, PyAny>) -> Option<Self> {
        match self {
            Self::Builtin(Builtin::Int) => key.cast_exact::<PyInt>().ok().map(Element::from),
            Self::Builtin(Builtin::String) => key.cast_exact::<PyString>().ok().map(Element::from),
            Self::Builtin(Builtin::Float) if key.is_exact_instance_of::<PyFloat>() => Some(self),
            Self::Object(Object::Unsafe { ty, .. }) if std::ptr::eq(key.get_type_ptr(), ty) => {
                Some(self)
            }
            _ => None,
        }
    }
}
impl From<&Bound<'_, PyInt>> for Element {
    fn from(int: &Bound<'_, PyInt>) -> Self {
        if int.is_compact() {
            Self::Builtin(Builtin::Int)
        } else {
            Self::from_object(int)
        }
    }
}
impl From<&Bound<'_, PyString>> for Element {
    fn from(string: &Bound<'_, PyString>) -> Self {
        if matches!(string.kind(), UnicodeByteKind::One) {
            Self::Builtin(Builtin::String)
        } else {
            Self::from_object(string)
        }
    }
}
impl From<&Bound<'_, PyAny>> for Element {
    fn from(item: &Bound<'_, PyAny>) -> Self {
        try_cast! {
            match item {
                CaseExact::PyInt(int) => int.into(),
                CaseExact::PyString(s) => s.into(),
                CaseExact::PyFloat(_) => Builtin::Float.into(),
                _ => Self::from_object(item),
            }
        }
    }
}
/// What builtin type is compared?
#[derive(Debug, Clone, Copy)]
pub(super) enum Builtin {
    /// Latin string compare: safe for any two latin (one byte per char) strings.
    String,
    /// Bounded int compare: compare any two longs that fit in a single machine word.
    Int,
    /// Float compare: compare any two floats.
    Float,
}
/// Is the generic object comparison done on same-type objects or not?
#[derive(Debug, Clone, Copy, From)]
pub(super) enum Object {
    /// Homogeneous compare: safe for any two comparable objects of the same type
    Unsafe {
        /// Plain type pointer: the objects in the sort buffer keep their types alive.
        ty: *mut ffi::PyTypeObject,
        func: ffi::richcmpfunc,
    },
    /// Heterogeneous compare: default, always safe to fall back on.
    Safe,
}
impl From<*mut ffi::PyTypeObject> for Object {
    fn from(ty: *mut ffi::PyTypeObject) -> Self {
        match unsafe { (*ty).tp_richcompare } {
            Some(func) => Self::Unsafe { ty, func },
            None => Self::Safe,
        }
    }
}
