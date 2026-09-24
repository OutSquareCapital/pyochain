use std::sync::{
    Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError,
};
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
}
// TODO: rename the method of MutexExt and RwLockExt to better match what they do.

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
}
pub trait RwLockExtMethods<T> {
    /// Read the inner value of the `RwLock`.\
    /// If the lock is poisoned, it will return the inner value of said lock.
    fn read_or_inner(&self) -> RwLockReadGuard<'_, T>;
    fn write_or_inner(&self) -> RwLockWriteGuard<'_, T>;
    /// Tries to acquire the write lock and returns the inner value.\
    /// If the lock is poisoned, it will return the inner value of the poisoned lock.
    /// If the lock is already held (by this or another thread), it will panic.
    fn try_write_or_inner(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T> RwLockExtMethods<T> for RwLock<T> {
    #[inline(always)]
    fn read_or_inner(&self) -> RwLockReadGuard<'_, T> {
        std::sync::RwLock::read(self).unwrap_or_else(PoisonError::into_inner)
    }
    #[inline(always)]
    fn write_or_inner(&self) -> RwLockWriteGuard<'_, T> {
        std::sync::RwLock::write(self).unwrap_or_else(PoisonError::into_inner)
    }
    #[inline(always)]
    fn try_write_or_inner(&self) -> RwLockWriteGuard<'_, T> {
        match self.try_write() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => panic!("data already locked - reentrant bug"),
        }
    }
}
