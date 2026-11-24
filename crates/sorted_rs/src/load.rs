use derive_more::Deref;
use pyo3::{exceptions::PyValueError, prelude::*};
use tap::Pipe;
#[derive(Clone, Copy, Debug, Deref)]
pub(super) struct Load(usize);
impl Default for Load {
    fn default() -> Self {
        Self::DEFAULT
    }
}
impl<'py> TryFrom<&Bound<'py, PyAny>> for Load {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        value
            .extract::<usize>()?
            .pipe(|value| (Self::MIN..=Self::MAX).contains(&value).then_some(value))
            .ok_or_else(|| {
                format!(
                    "Load must be at least {} and at most {}",
                    Self::MIN,
                    Self::MAX
                )
            })
            .map_err(PyValueError::new_err)
            .map(Self)
    }
}
impl Load {
    /// Keeps `max_size` within `u32`, so a sublist index always fits in the iterator cursor.
    const MAX: usize = u32::MAX as usize / 2;
    const MIN: usize = 1;
    const DEFAULT: Self = Self(1000);
    /// Maximum size of a sublist. If a sublist is larger than this, it will be split in two.
    pub const fn max_size(self) -> usize {
        self.0 * 2
    }
    /// Minimum size of a sublist. If a sublist is smaller than this, it will be merged with its neighbor.
    pub const fn min_size(self) -> usize {
        self.0 / 2
    }
}
