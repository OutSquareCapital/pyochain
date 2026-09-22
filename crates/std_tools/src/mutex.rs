use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
pub trait ArcExtMethods {
    fn is(&self, other: &Self) -> bool;
}
impl<T> ArcExtMethods for Arc<T> {
    fn is(&self, other: &Self) -> bool {
        Arc::ptr_eq(self, other)
    }
}

pub trait MutexExtMethods<T> {
    fn try_into_inner(&self) -> MutexGuard<'_, T>;
    fn map_lock<R, F: FnOnce(&T) -> R>(&self, f: F) -> R;
}

impl<T> MutexExtMethods<T> for Mutex<T> {
    /// Tries to acquire the lock and returns the inner value.\
    /// If the lock is poisoned, it will return the inner value of the poisoned lock.
    /// If the lock is already held by another thread, it will panic.
    #[inline(always)]
    fn try_into_inner(&self) -> MutexGuard<'_, T> {
        match self.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => panic!("data already locked - reentrant bug"),
        }
    }
    #[inline(always)]
    fn map_lock<R, F: FnOnce(&T) -> R>(&self, f: F) -> R {
        f(&self.try_into_inner())
    }
}
