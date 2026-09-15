use std::sync::{Mutex, MutexGuard, TryLockError};

pub trait MutexExtMethods<T> {
    fn try_into_inner(&self) -> MutexGuard<'_, T>;
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
}
