// Pure functions tools for pyochain
use crate::option::{PySome, get_none_singleton};
use crate::result::{PyErr as PyochainErr, PyOk, PyResultEnum};
use crate::types::PyClassInit;
use pyo3::types::{PyAny, PyBool, PyFunction};
use pyo3::{IntoPyObjectExt, prelude::*};

#[pymodule(name = "_tools")]
pub fn tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(try_find, m)?)?;
    Ok(())
}
#[pyfunction]
pub fn try_find(data: &Bound<'_, PyAny>, predicate: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
    let py = data.py();
    for item in data.try_iter()? {
        let val = item?;
        match predicate.call1((&val,))?.extract::<PyResultEnum<'_>>()? {
            PyResultEnum::Ok(ok_ref) => {
                if unsafe {
                    ok_ref
                        .get()
                        .value
                        .cast_bound_unchecked::<PyBool>(py)
                        .is_true()
                } {
                    let some_val = PySome::new(val.unbind()).init(py)?.into_any();
                    return Ok(PyOk::new(some_val).into_py_any(py)?);
                }
            }
            PyResultEnum::Err(err_ref) => {
                let err_val = err_ref.get().error.clone_ref(py);
                return Ok(PyochainErr::new(err_val).into_py_any(py)?);
            }
        }
    }
    let none = get_none_singleton(py)?;
    Ok(PyOk::new(none).into_py_any(py)?)
}
