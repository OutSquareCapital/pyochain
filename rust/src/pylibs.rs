use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyInt, PyIterator, PyList, PyTuple};
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

    use pyo3::types::PyDict;

    use crate::args::Concatenate;

    use super::*;

    const ITERTOOLS: &str = "itertools";
    static ACCUMULATE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static BATCHED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static TEE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static GROUP_BY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static ZIP_LONGEST: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static FILTER_FALSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static MAP_STAR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static COUNT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static TAKE_WHILE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static DROP_WHILE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static COMBINATIONS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static COMBINATIONS_WITH_REPLACEMENT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static COMPRESS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static CYCLE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static PAIRWISE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static PERMUTATIONS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
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
    pub fn take_while<'py>(
        predicate: &Bound<'py, PyAny>,
        iterator: &Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        TAKE_WHILE
            .import(iterator.py(), ITERTOOLS, "takewhile")?
            .call1((predicate, iterator))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn drop_while<'py>(
        predicate: &Bound<'py, PyAny>,
        iterator: &Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        DROP_WHILE
            .import(iterator.py(), ITERTOOLS, "dropwhile")?
            .call1((predicate, iterator))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn accumulate<'py>(
        iterator: &Bound<'py, PyIterator>,
        func: Option<Bound<'py, PyAny>>,
        initial: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "initial"), initial)?;
        ACCUMULATE
            .import(iterator.py(), ITERTOOLS, "accumulate")?
            .call((iterator, func), Some(&kwargs))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn batched<'py>(
        iterator: &Bound<'py, PyIterator>,
        n: &Bound<'py, PyInt>,
        strict: &bool,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "strict"), strict)?;
        BATCHED
            .import(iterator.py(), ITERTOOLS, "batched")?
            .call((iterator, n), Some(&kwargs))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn compress<'py>(
        iterator: &Bound<'py, PyIterator>,
        selectors: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        COMPRESS
            .import(iterator.py(), ITERTOOLS, "compress")?
            .call1((iterator, selectors))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn cycle<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyIterator>> {
        CYCLE
            .import(iterator.py(), ITERTOOLS, "cycle")?
            .call1((iterator,))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn permutations<'py>(
        iterator: &Bound<'py, PyIterator>,
        r: Option<usize>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        PERMUTATIONS
            .import(iterator.py(), ITERTOOLS, "permutations")?
            .call1((iterator, r))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn combinations<'py>(
        iterator: &Bound<'py, PyIterator>,
        r: &Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        COMBINATIONS
            .import(iterator.py(), ITERTOOLS, "combinations")?
            .call1((iterator, r))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn combinations_with_replacement<'py>(
        iterator: &Bound<'py, PyIterator>,
        r: &Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        COMBINATIONS_WITH_REPLACEMENT
            .import(iterator.py(), ITERTOOLS, "combinations_with_replacement")?
            .call1((iterator, r))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn filter_false<'py>(
        func: Option<Bound<'py, PyAny>>,
        iterator: &Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        FILTER_FALSE
            .import(iterator.py(), ITERTOOLS, "filterfalse")?
            .call1((func, iterator))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn group_by<'py>(
        iterator: &Bound<'py, PyIterator>,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        GROUP_BY
            .import(iterator.py(), ITERTOOLS, "groupby")?
            .call1((iterator, key))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn pairwise<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyIterator>> {
        PAIRWISE
            .import(iterator.py(), ITERTOOLS, "pairwise")?
            .call1((iterator,))
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

    #[inline(always)]
    pub fn zip_longest<'py>(
        iterator: &Bound<'py, PyIterator>,
        others: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        ZIP_LONGEST
            .import(py, ITERTOOLS, "zip_longest")?
            .concat1(iterator, others)
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
