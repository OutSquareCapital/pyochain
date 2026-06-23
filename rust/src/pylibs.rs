use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyIterator, PyList, PyTuple};
use pyo3::{intern, prelude::*};
use tap::prelude::*;

/// Built-in Python functions and objects
pub mod builtins {
    use super::*;

    const BUILTINS: &str = "builtins";

    static OBJECT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static ALL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static ANY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    /// Create a unique sentinel object
    #[inline(always)]
    pub fn sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        OBJECT.import(py, BUILTINS, "object")?.call0()?.pipe(Ok)
    }
    #[inline(always)]
    pub fn all(iterator: Bound<'_, PyIterator>) -> PyResult<Bound<'_, PyBool>> {
        ALL.import(iterator.py(), BUILTINS, "all")?
            .call1((iterator,))?
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
            .pipe(Ok)
    }
    #[inline(always)]
    pub fn any(iterator: Bound<'_, PyIterator>) -> PyResult<Bound<'_, PyBool>> {
        ANY.import(iterator.py(), BUILTINS, "any")?
            .call1((iterator,))?
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
            .pipe(Ok)
    }
}

/// Python itertools module functions and objects
pub mod itertools {
    use super::*;

    const ITERTOOLS: &str = "itertools";
    static TEE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static GROUP_BY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    #[inline(always)]
    pub fn tee<'py>(
        iterator: Bound<'py, PyIterator>,
        n: Option<usize>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        TEE.import(iterator.py(), ITERTOOLS, "tee")?
            .call1((iterator, n.unwrap_or(2)))?
            .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyTuple>() })
            .pipe(Ok)
    }
    #[inline(always)]
    pub fn group_by<'py>(
        iterator: Bound<'py, PyIterator>,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        GROUP_BY
            .import(iterator.py(), ITERTOOLS, "groupby")?
            .call1((iterator, key))?
            .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
            .pipe(Ok)
    }
}
pub mod pyochain {
    use super::*;
    const PYOCHAIN: &str = "pyochain";
    pub mod vec {
        use super::*;
        static VEC: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        #[inline(always)]
        pub fn from_ref<'py>(obj: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyAny>> {
            let py = obj.py();
            VEC.import(py, PYOCHAIN, "Vec")?
                .getattr(intern!(py, "from_ref"))?
                .call1((obj,))
        }
    }
    pub mod iter {
        use super::*;
        static ITER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        #[inline(always)]
        pub fn new<'py>(iterable: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyIterator>> {
            ITER.import(iterable.py(), PYOCHAIN, "Iter")?
                .call1((iterable,))?
                .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
                .pipe(Ok)
        }
        #[inline(always)]
        pub fn once<'py>(val: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyIterator>> {
            let py = val.py();
            ITER.import(py, PYOCHAIN, "Iter")?
                .getattr(intern!(py, "once"))?
                .call1((val,))?
                .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
                .pipe(Ok)
        }
    }
}
