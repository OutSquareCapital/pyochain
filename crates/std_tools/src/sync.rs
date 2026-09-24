use std::sync;
pub trait ArcExtMethods {
    fn is(&self, other: &Self) -> bool;
}
impl<T> ArcExtMethods for sync::Arc<T> {
    fn is(&self, other: &Self) -> bool {
        sync::Arc::ptr_eq(self, other)
    }
}

pub trait MutexExtMethods<T> {
    fn lock_or_inner(&self) -> sync::MutexGuard<'_, T>;
}

impl<T> MutexExtMethods<T> for sync::Mutex<T> {
    /// Tries to acquire the lock and returns the inner value.\
    /// If the lock is poisoned, it will return the inner value of the poisoned lock.
    /// If the lock is already held by another thread, it will panic.
    #[inline(always)]
    fn lock_or_inner(&self) -> sync::MutexGuard<'_, T> {
        #[cfg(debug_assertions)]
        return ok_or_block(self.try_lock());
        #[cfg(not(debug_assertions))]
        return self.lock().unwrap();
    }
}
pub trait RwLockExtMethods<T> {
    /// Read the inner value of the `RwLock`.\
    /// If the lock is poisoned, it will return the inner value of said lock.
    fn read_or_inner(&self) -> sync::RwLockReadGuard<'_, T>;
    /// Tries to acquire the write lock and returns the inner value.\
    /// If the lock is poisoned, it will return the inner value of the poisoned lock.
    /// If the lock is already held (by this or another thread), it will panic.
    fn write_or_inner(&self) -> sync::RwLockWriteGuard<'_, T>;
}

impl<T> RwLockExtMethods<T> for sync::RwLock<T> {
    #[inline(always)]
    fn read_or_inner(&self) -> sync::RwLockReadGuard<'_, T> {
        #[cfg(debug_assertions)]
        return self.read().unwrap_or_else(sync::PoisonError::into_inner);
        #[cfg(not(debug_assertions))]
        return self.read().unwrap();
    }
    #[inline(always)]
    fn write_or_inner(&self) -> sync::RwLockWriteGuard<'_, T> {
        #[cfg(debug_assertions)]
        return ok_or_block(self.try_write());
        #[cfg(not(debug_assertions))]
        return self.write().unwrap();
    }
}
#[cfg(debug_assertions)]
#[inline(always)]
fn ok_or_block<T>(res: Result<T, sync::TryLockError<T>>) -> T {
    match res {
        Ok(guard) => guard,
        Err(sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(sync::TryLockError::WouldBlock) => panic!("data already locked - reentrant bug"),
    }
}
