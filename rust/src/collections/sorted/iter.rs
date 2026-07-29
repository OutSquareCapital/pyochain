use pyo3::{prelude::*, types::PyList};
#[inline(always)]
pub fn try_iterator_into_list<'py, T: Sized + IntoPyObject<'py>>(
    acc: Bound<'py, PyList>,
    x: PyResult<T>,
) -> PyResult<Bound<'py, PyList>> {
    acc.append(x?)?;
    Ok(acc)
}
#[inline(always)]
pub fn iterator_into_list<'py, T: Sized + IntoPyObject<'py>>(
    acc: Bound<'py, PyList>,
    x: T,
) -> PyResult<Bound<'py, PyList>> {
    acc.append(x)?;
    Ok(acc)
}
