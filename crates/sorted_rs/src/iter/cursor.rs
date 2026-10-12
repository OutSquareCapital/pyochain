use std::sync::atomic::{AtomicU64, Ordering};

use tap::prelude::*;

use crate::Loc;
#[derive(Debug, Default)]
pub(super) struct Cursor(AtomicU64);
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
