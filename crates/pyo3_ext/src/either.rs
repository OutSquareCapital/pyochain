use either::Either;
use pyo3::prelude::*;

use crate::types::BoundedEither;
pub trait EitherExtMethods<'py> {
    fn py(&self) -> Python<'py>;
}
impl<'py, T, U> EitherExtMethods<'py> for BoundedEither<'py, T, U> {
    fn py(&self) -> Python<'py> {
        match self {
            Either::Left(x) => x.py(),
            Either::Right(x) => x.py(),
        }
    }
}
