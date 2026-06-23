use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyIterator, PyList, PyTuple};
use pyo3::{intern, prelude::*};

/// Built-in Python functions and objects
pub mod builtins {
    use super::*;

    const BUILTINS: &str = "builtins";

    static OBJECT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static ALL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static ANY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static MAP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static FILTER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    /// Create a unique sentinel object
    #[inline(always)]
    pub fn sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        OBJECT.import(py, BUILTINS, "object")?.call0()
    }
    #[inline(always)]
    pub fn all(iterator: Bound<'_, PyIterator>) -> PyResult<Bound<'_, PyBool>> {
        ALL.import(iterator.py(), BUILTINS, "all")?
            .call1((iterator,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
    }
    #[inline(always)]
    pub fn any(iterator: Bound<'_, PyIterator>) -> PyResult<Bound<'_, PyBool>> {
        ANY.import(iterator.py(), BUILTINS, "any")?
            .call1((iterator,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
    }
    #[inline(always)]
    pub fn map<'py>(
        func: Bound<'py, PyAny>,
        iterator: Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        MAP.import(iterator.py(), BUILTINS, "map")?
            .call1((func, iterator))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn filter<'py>(
        func: Option<Bound<'py, PyAny>>,
        iterator: Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        FILTER
            .import(iterator.py(), BUILTINS, "filter")?
            .call1((func, iterator))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
    }
}

/// Python itertools module functions and objects
pub mod itertools {

    use crate::args::Concatenate;

    use super::*;

    const ITERTOOLS: &str = "itertools";
    static TEE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static GROUP_BY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static ZIP_LONGEST: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static MAP_STAR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static COUNT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    #[inline(always)]
    pub fn count<'py>(
        py: Python<'py>,
        start: &i32,
        step: &i32,
    ) -> PyResult<Bound<'py, PyIterator>> {
        COUNT
            .import(py, ITERTOOLS, "count")?
            .call1((start, step))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }

    #[inline(always)]
    pub fn tee<'py>(
        iterator: Bound<'py, PyIterator>,
        n: Option<usize>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        TEE.import(iterator.py(), ITERTOOLS, "tee")?
            .call1((iterator, n.unwrap_or(2)))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyTuple>() })
    }
    #[inline(always)]
    pub fn group_by<'py>(
        iterator: Bound<'py, PyIterator>,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        GROUP_BY
            .import(iterator.py(), ITERTOOLS, "groupby")?
            .call1((iterator, key))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn zip_longest<'py>(
        iterator: Bound<'py, PyIterator>,
        others: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        ZIP_LONGEST
            .import(py, ITERTOOLS, "zip_longest")?
            .concat1(&iterator, others)
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn map_star<'py>(
        func: Bound<'py, PyAny>,
        iterable: Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        MAP_STAR
            .import(iterable.py(), ITERTOOLS, "starmap")?
            .call1((func, iterable))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
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
                .call1((iterable,))
                .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
        }
        #[inline(always)]
        pub fn once<'py>(val: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyIterator>> {
            let py = val.py();
            ITER.import(py, PYOCHAIN, "Iter")?
                .getattr(intern!(py, "once"))?
                .call1((val,))
                .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
        }
    }
}
