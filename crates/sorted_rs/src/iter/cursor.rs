use std::sync::atomic::{AtomicU64, Ordering};

use tap::prelude::*;

use crate::Loc;
#[derive(Debug, Default)]
pub(super) struct Cursor(AtomicU64);
impl Cursor {
    #[inline(always)]
    pub fn shift_fwd(pos: usize) -> u64 {
        (pos as u64 + 1) << u32::BITS
    }
    #[inline(always)]
    pub fn shift_rev<T>(pos: usize, v: &[T]) -> u64 {
        ((pos as u64) << u32::BITS) | (v.len() - 1) as u64
    }
    #[allow(clippy::cast_possible_truncation)]
    #[inline(always)]
    pub fn load(&self) -> (u64, usize, usize) {
        let at = self.0.load(Ordering::Relaxed);
        (
            at,
            (at >> u32::BITS) as usize,
            (at & u64::from(u32::MAX)) as usize,
        )
    }
    #[inline(always)]
    pub fn store(&self, at: u64) {
        self.0.store(at, Ordering::Relaxed);
    }
}

impl From<Loc> for Cursor {
    fn from(loc: Loc) -> Self {
        loc.conv::<u64>().conv::<AtomicU64>().pipe(Self)
    }
}
