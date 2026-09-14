use either::Either;
use pyo3::prelude::*;

pub trait EitherExtMethods<'py> {
    fn py(&self) -> Python<'py>;
}
impl<'py, T, U> EitherExtMethods<'py> for Either<Bound<'py, T>, Bound<'py, U>> {
    fn py(&self) -> Python<'py> {
        match self {
            Either::Left(x) => x.py(),
            Either::Right(x) => x.py(),
        }
    }
}
