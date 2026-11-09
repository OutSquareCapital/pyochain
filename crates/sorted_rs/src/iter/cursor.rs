use std::{
    ops::Deref,
    sync::atomic::{AtomicU64, Ordering},
};

use derive_more::{Constructor, Deref};
use tap::prelude::*;

use crate::bounds::Loc;
#[derive(Debug, Default, Constructor, Deref)]
pub struct Cursor(AtomicU64);
impl Cursor {
    pub fn into_bounded(self, limit: u64) -> BoundedCursor {
        BoundedCursor::new(self, limit)
    }
}

#[derive(Debug, Default, Constructor, Deref)]
pub struct BoundedCursor {
    #[deref]
    cursor: Cursor,
    pub(super) limit: u64,
}
impl BoundedCursor {
    pub fn swap(&self) -> Self {
        self.limit
            .conv::<AtomicU64>()
            .pipe(Cursor::new)
            .into_bounded(self.cursor.load().0)
    }
}
pub trait CursorMethods {
    #[allow(clippy::cast_possible_truncation)]
    fn load(&self) -> (u64, usize, usize);
    fn store(&self, at: u64);
}
impl CursorMethods for Cursor {
    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    fn load(&self) -> (u64, usize, usize) {
        let at = self.0.load(Ordering::Relaxed);
        let pos = (at >> u32::BITS) as usize;
        let idx = (at & u64::from(u32::MAX)) as usize;
        (at, pos, idx)
    }
    #[inline(always)]
    fn store(&self, at: u64) {
        self.0.store(at, Ordering::Relaxed);
    }
}
impl Cursor {
    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    pub fn load(&self) -> (u64, usize, usize) {
        let at = self.0.load(Ordering::Relaxed);
        let pos = (at >> u32::BITS) as usize;
        let idx = (at & u64::from(u32::MAX)) as usize;
        (at, pos, idx)
    }
    #[inline(always)]
    pub fn store(&self, at: u64) {
        self.0.store(at, Ordering::Relaxed);
    }
    #[inline(always)]
    pub fn new_pos(pos: usize) -> u64 {
        (pos as u64) << u32::BITS
    }
}
impl<T> CursorMethods for T
where
    T: Deref<Target = Cursor>,
{
    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    fn load(&self) -> (u64, usize, usize) {
        self.deref().load()
    }
    #[inline(always)]
    fn store(&self, at: u64) {
        self.deref().store(at);
    }
}

impl From<Loc> for Cursor {
    fn from(loc: Loc) -> Self {
        loc.conv::<u64>().conv::<AtomicU64>().pipe(Self)
    }
}
impl From<Loc> for u64 {
    fn from(loc: Loc) -> Self {
        (loc.pos as Self) << u32::BITS | loc.idx as Self
    }
}
