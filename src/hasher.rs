use pyo3::{PyClass, prelude::*, types::DerefToPyAny};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
pub fn hash_fn(tag: u8, value: isize) -> u64 {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    value.hash(&mut hasher);
    hasher.finish()
}
/// Mirrors `_collections_abc.Set._hash`.\
/// Python masks with `MASK = 2 * sys.maxsize + 1` (i.e. the full 64-bit range) after every step.\
/// This is equivalent to doing the whole computation with wrapping u64 arithmetic,\
/// and reinterpreting the final bit pattern as a signed 64-bit hash.
pub fn set_hash<T: PyClass + DerefToPyAny>(slf: &Bound<'_, T>) -> PyResult<isize> {
    let length = slf.len()?;
    let mut h = 1_927_868_237_usize.wrapping_mul(length.wrapping_add(1));
    for x in slf.try_iter()? {
        let hx = x?.hash()?.cast_unsigned();
        let mixed = hx ^ (hx << 16) ^ 0x055b_4db3;
        h ^= mixed.wrapping_mul(3_644_798_167_usize);
    }
    h ^= (h >> 11) ^ (h >> 25);
    h = h.wrapping_mul(69_069).wrapping_add(907_133_923);
    let h = h.cast_signed();
    Ok(if h == -1 { 590_923_713 } else { h })
}
