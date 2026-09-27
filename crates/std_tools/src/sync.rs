use std::sync;
pub trait ArcExtMethods {
    fn is(&self, other: &Self) -> bool;
}
impl<T> ArcExtMethods for sync::Arc<T> {
    fn is(&self, other: &Self) -> bool {
        sync::Arc::ptr_eq(self, other)
    }
}
