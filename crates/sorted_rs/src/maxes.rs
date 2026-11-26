use derive_more::{Deref, DerefMut, From, IntoIterator};
use pyo3::prelude::*;

use crate::types::VecPy;
#[derive(Deref, DerefMut, From, Debug, Default, IntoIterator)]
pub struct Maxes(VecPy);
impl Maxes {
    pub(super) fn clone_ref(&self, py: Python<'_>) -> Self {
        self.iter()
            .map(|x| x.clone_ref(py))
            .collect::<VecPy>()
            .into()
    }
}
