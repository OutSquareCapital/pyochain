use pyo3::{
    PyTypeInfo,
    prelude::*,
    sync::PyOnceLock,
    types::{DerefToPyAny, PyDict, PyString},
};

static PFORMAT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

pub fn pformat<'py, T: Sized + IntoPyObject<'py>>(
    py: Python<'py>,
    obj: T,
    sort_dicts: bool,
) -> PyResult<Bound<'py, PyString>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("sort_dicts", sort_dicts).unwrap();
    PFORMAT
        .import(py, "pprint", "pformat")?
        .call((obj,), Some(&kwargs))
        .map(|x| unsafe { x.cast_into_unchecked::<PyString>() })
}

pub fn get_repr<T: Sized + PyTypeInfo + DerefToPyAny>(
    obj: Bound<'_, T>,
) -> PyResult<Bound<'_, PyString>> {
    let py = obj.py();
    let length = obj.len()?;

    match length {
        0 => Ok(PyString::new(py, "")),
        _ => pformat(py, obj, false).map(|x| {
            let full = x.to_str().unwrap();
            PyString::new(py, &full[1..full.len() - 1])
        }),
    }
}
