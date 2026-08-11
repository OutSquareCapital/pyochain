use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
pub fn hash_fn(tag: u8, value: isize) -> u64 {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    value.hash(&mut hasher);
    hasher.finish()
}
