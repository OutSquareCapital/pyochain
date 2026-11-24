pub trait WithCapacity<T>: FromIterator<T> {
    fn with_capacity(capacity: usize) -> Self;
    fn push(&mut self, item: T);
}
impl<T> WithCapacity<T> for Vec<T> {
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }
    fn push(&mut self, item: T) {
        self.push(item);
    }
}
