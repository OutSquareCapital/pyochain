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

use crate::item::Item;

use super::keys::SortKey;

/// The actual, high level dispatch type for the sort strategy.\
/// The two variants simply answer: "Is the comparison done on a tuple or a single element?"
#[derive(Debug, Clone, Copy)]
pub(super) enum SortBy {
    Tuple(Element),
    Element(Element),
}
/// `CPython`'s pre-sort check: the properties of the keys, proven on every item.\
/// The compare strategy is only chosen once they are all known, by converting into [`SortBy`].
#[derive(Debug, Clone, Copy)]
pub(super) struct KeysCheck {
    /// Type of the first key, assumed to be representative of the whole list.
    key_type: *mut ffi::PyTypeObject,
    /// Compare to use if every key is proven to fit it.
    kind: Element,
    in_tuples: bool,
    same_type: bool,
    /// The builtin fast path of `kind` still holds: bounded ints, or latin strings.
    builtin_fits: bool,
}
impl KeysCheck {
    /// Assumes the first item is representative of the whole list.\
    /// It still has to go through [`check`](Self::check) like every other item.
    pub fn new(first: &Bound<'_, PyAny>) -> Self {
        let first_key = try_get_first(first);
        let key = first_key.unwrap_or(first);
        Self {
            key_type: key.get_type_ptr(),
            // Tuples of tuples: the first items are compared with the safe compare.
            kind: if first_key.is_some() && key.is_exact_instance_of::<PyTuple>() {
                Element::SAFE
            } else {
                Element::from_key(key)
            },
            in_tuples: first_key.is_some(),
            same_type: true,
            builtin_fits: true,
        }
    }
    /// Proves the assumption on `item`.\
    /// For non-tuples, nothing is left to check after a type mismatch.
    /// For tuples, every item must still be checked to be a non-empty tuple.
    #[inline(always)]
    pub fn check(&mut self, item: &Bound<'_, PyAny>) {
        if self.in_tuples {
            if let Some(key) = try_get_first(item) {
                self.check_key(key);
            } else {
                self.in_tuples = false;
                self.same_type = false;
            }
        } else if self.same_type {
            self.check_key(item);
        }
    }
    #[inline(always)]
    fn check_key(&mut self, key: &Bound<'_, PyAny>) {
        if !std::ptr::eq(key.get_type_ptr(), self.key_type) {
            self.same_type = false;
        } else if self.same_type {
            // SAFETY: `key` is exactly of type `key_type`, which `kind` was chosen from.
            match self.kind {
                Element::Builtin(Builtin::Int) => {
                    self.builtin_fits &= unsafe { key.cast_unchecked::<PyInt>() }.is_compact();
                }
                Element::Builtin(Builtin::String) => {
                    self.builtin_fits &= matches!(
                        unsafe { key.cast_unchecked::<PyString>() }.kind(),
                        UnicodeByteKind::One
                    );
                }
                _ => {}
            }
        }
    }

    /// Citation from `CPython`'s listobject.c:\
    /// > Choose the best compare, given what we now know about the keys.
    pub fn into_strategy(self) -> SortBy {
        let element = match self.kind {
            _ if !self.same_type => Element::SAFE,
            Element::Builtin(Builtin::Int | Builtin::String) if !self.builtin_fits => {
                self.key_type.conv::<Object>().into()
            }
            kind => kind,
        };
        if self.in_tuples {
            SortBy::Tuple(element)
        } else {
            SortBy::Element(element)
        }
    }
}
impl SortBy {
    pub(super) fn sort<T: SortKey>(self, mut items: Vec<T>) -> PyResult<Vec<T>> {
        use Builtin as Bl;
        use Element as El;
        use Object as Ob;
        use SortBy as Sb;
        let mut try_cmp = TryCmp::default();
        match self {
            Sb::Element(El::Object(Ob::Safe)) => {
                items.sort_by(|a, b| try_cmp.catch(|| a.cmp(b)));
            }
            Sb::Element(El::Builtin(Bl::Int)) => items.sort_by(SortKey::long),
            Sb::Element(El::Builtin(Bl::Float)) => items.sort_by(SortKey::float),
            Sb::Element(El::Builtin(Bl::String)) => items.sort_by(SortKey::latin),
            Sb::Element(El::Object(Ob::Unsafe(func))) => {
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
                items.sort_by(|a, b| try_cmp.catch(|| a.try_tuple(b, Item::cmp)));
            }
            Sb::Tuple(El::Object(Ob::Unsafe(func))) => {
                items.sort_by(|a, b| {
                    try_cmp.catch(|| a.try_tuple(b, |a, b| SortKey::unsafe_object(a, b, func)))
                });
            }
        }
        try_cmp.0.map_or(Ok(items), Err)
    }
}
/// First item of a non-empty exact tuple.
#[inline(always)]
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
    /// Compare for keys of exactly this type, before proving it on all of them.
    fn from_key(key: &Bound<'_, PyAny>) -> Self {
        try_cast! {
            match key {
                CaseExact::PyInt(_) => Builtin::Int.into(),
                CaseExact::PyString(_) => Builtin::String.into(),
                CaseExact::PyFloat(_) => Builtin::Float.into(),
                _ => key.get_type_ptr().conv::<Object>().into(),
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
    Unsafe(ffi::richcmpfunc),
    /// Heterogeneous compare: default, always safe to fall back on.
    Safe,
}
impl From<*mut ffi::PyTypeObject> for Object {
    fn from(ty: *mut ffi::PyTypeObject) -> Self {
        match unsafe { (*ty).tp_richcompare } {
            Some(func) => Self::Unsafe(func),
            None => Self::Safe,
        }
    }
}
