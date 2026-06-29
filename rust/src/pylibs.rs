use crate::args::{Args, Concatenate};
/// This module contains Python built-in functions and objects, as well as functions and objects from the `itertools` and `functools` modules.
/// Each submodule declares a const string with the name of the module, and a const `PyOnceLock` + associated fn for each function or object that is imported from that module.
/// This pattern ensure maximum performance by only importing the function or object once, and reusing it for subsequent calls.
/// We also use unsafe casts to correct types, aggressive inlining, and `&Bound` to maximize performance.
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyInt, PyIterator, PyList, PyNone, PySequence, PyTuple};
use pyo3::{intern, prelude::*};
use tap::prelude::*;

/// Python `builtins` functions and objects
pub mod builtins {
    use super::*;

    const BUILTINS: &str = "builtins";

    const OBJECT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const ALL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const ANY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const MAX: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const MIN: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const SUM: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const ENUMERATE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const MAP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const FILTER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const SORTED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const ZIP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    /// Create a unique sentinel object. Equivalent to `object()` in Python. On >=3.15, this will become unneded thanks to the new sentinel builtin.
    #[inline(always)]
    pub fn sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        OBJECT.import(py, BUILTINS, "object")?.call0()
    }
    #[inline(always)]
    pub fn all<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyBool>> {
        ALL.import(iterator.py(), BUILTINS, "all")?
            .call1((iterator,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
    }
    #[inline(always)]
    pub fn any<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyBool>> {
        ANY.import(iterator.py(), BUILTINS, "any")?
            .call1((iterator,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
    }

    pub fn enumerate<'py>(
        iterator: &Bound<'py, PyIterator>,
        start: usize,
    ) -> PyResult<Bound<'py, PyIterator>> {
        ENUMERATE
            .import(iterator.py(), BUILTINS, "enumerate")?
            .call1((iterator, start))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
    }

    #[inline(always)]
    pub fn filter<'py>(
        func: Option<&Bound<'py, PyAny>>,
        iterator: &Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        FILTER
            .import(iterator.py(), BUILTINS, "filter")?
            .call1((func, iterator))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn max<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyAny>> {
        MAX.import(iterator.py(), BUILTINS, "max")?
            .call1((iterator,))
    }
    #[inline(always)]
    pub fn max_by<'py>(
        iterator: &Bound<'py, PyIterator>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(iterator.py());
        kwargs.set_item(intern!(iterator.py(), "key"), key)?;
        MAX.import(iterator.py(), BUILTINS, "max")?
            .call((iterator,), Some(&kwargs))
    }
    #[inline(always)]
    pub fn min<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyAny>> {
        MIN.import(iterator.py(), BUILTINS, "min")?
            .call1((iterator,))
    }
    #[inline(always)]
    pub fn min_by<'py>(
        iterator: &Bound<'py, PyIterator>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(iterator.py());
        kwargs.set_item(intern!(iterator.py(), "key"), key)?;
        MIN.import(iterator.py(), BUILTINS, "min")?
            .call((iterator,), Some(&kwargs))
    }
    #[inline(always)]
    pub fn sum<'py>(iterator: &Bound<'py, PyIterator>, start: &i32) -> PyResult<Bound<'py, PyAny>> {
        SUM.import(iterator.py(), BUILTINS, "sum")?
            .call1((iterator, start))
    }
    #[inline(always)]
    pub fn map<'py>(
        func: &Bound<'py, PyAny>,
        iterator: &Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        MAP.import(iterator.py(), BUILTINS, "map")?
            .call1((func, iterator))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
    }
    /// first arg is a function, the rest is a variable number of iterables.
    #[inline(always)]
    pub fn map_with<'py>(args: &Args<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let py = args.py();
        MAP.import(py, BUILTINS, "map")?
            .call1(args)
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn sorted<'py>(
        iterator: &Bound<'py, PyIterator>,
        reverse: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let py = iterator.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "reverse"), reverse)?;
        SORTED
            .import(py, BUILTINS, "sorted")?
            .call((iterator,), Some(&kwargs))
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
    }
    #[inline(always)]
    pub fn sorted_by<'py>(
        iterator: &Bound<'py, PyIterator>,
        reverse: bool,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let py = iterator.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "reverse"), reverse)?;
        kwargs.set_item(intern!(py, "key"), key)?;
        SORTED
            .import(py, BUILTINS, "sorted")?
            .call((iterator,), Some(&kwargs))
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
    }
    #[inline(always)]
    pub fn zip<'py>(
        iterator: &Bound<'py, PyIterator>,
        others: &Args<'py>,
        strict: bool,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "strict"), strict)?;
        ZIP.import(py, BUILTINS, "zip")?
            .concat(iterator, others, Some(&kwargs))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
    }
}

/// Python `itertools` module functions and objects
pub mod itertools {

    use super::*;

    const ITERTOOLS: &str = "itertools";
    const ACCUMULATE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const BATCHED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const TEE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const GROUP_BY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const ZIP_LONGEST: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const FILTER_FALSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const MAP_STAR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const COUNT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const TAKE_WHILE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const DROP_WHILE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const COMBINATIONS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const COMBINATIONS_WITH_REPLACEMENT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const COMPRESS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const CYCLE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const PAIRWISE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const PRODUCT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const PERMUTATIONS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const REPEAT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    const ISLICE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    /// `itertools::chain` class.
    pub mod chain {
        use super::*;
        const CHAIN: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

        #[inline(always)]
        pub fn new<'py>(iterables: &Args<'py>) -> PyResult<Bound<'py, PyIterator>> {
            let py = iterables.py();
            CHAIN
                .import(py, ITERTOOLS, "chain")?
                .call1(iterables)
                .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
        }

        #[inline(always)]
        pub fn from_iterable<'py>(
            iterable: &Bound<'py, PyIterator>,
        ) -> PyResult<Bound<'py, PyIterator>> {
            let py = iterable.py();
            CHAIN
                .import(py, ITERTOOLS, "chain")?
                .getattr(intern!(py, "from_iterable"))?
                .call1((iterable,))
                .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
        }
    }
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
    pub fn tee<'py>(iterator: Bound<'py, PyIterator>, n: usize) -> PyResult<Bound<'py, PyTuple>> {
        TEE.import(iterator.py(), ITERTOOLS, "tee")?
            .call1((iterator, n))
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
            .import(py, ITERTOOLS, "accumulate")?
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
            .import(py, ITERTOOLS, "batched")?
            .call((iterator, n), Some(&kwargs))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn compress<'py>(
        iterator: &Bound<'py, PyIterator>,
        selectors: &Args<'py>,
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
    pub fn product<'py>(iterables: &Args<'py>, repeat: usize) -> PyResult<Bound<'py, PyIterator>> {
        let kwargs = PyDict::new(iterables.py());
        kwargs.set_item(intern!(iterables.py(), "repeat"), repeat)?;
        PRODUCT
            .import(iterables.py(), ITERTOOLS, "product")?
            .call(iterables, Some(&kwargs))
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
    pub fn nth<'py>(
        iterator: &Bound<'py, PyIterator>,
        n: usize,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = iterator.py();
        ISLICE
            .import(py, ITERTOOLS, "islice")?
            .call1((iterator, n, n + 1))?
            .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
            .next()
            .transpose()
    }
    #[inline(always)]
    pub fn slice<'py>(
        iterator: &Bound<'py, PyIterator>,
        start: &Option<&Bound<'py, PyInt>>,
        stop: &Option<&Bound<'py, PyInt>>,
        step: &Option<&Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        ISLICE
            .import(py, ITERTOOLS, "islice")?
            .call1((iterator, start, stop, step))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn skip<'py>(
        iterator: &Bound<'py, PyIterator>,
        n: &Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        ISLICE
            .import(py, ITERTOOLS, "islice")?
            .call1((iterator, n, PyNone::get(py)))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }

    #[inline(always)]
    pub fn zip_longest<'py>(
        iterator: &Bound<'py, PyIterator>,
        others: &Args<'py>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        ZIP_LONGEST
            .import(py, ITERTOOLS, "zip_longest")?
            .concat1(iterator, others)
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn repeat<'py>(
        obj: &Bound<'py, PyAny>,
        n: Option<&Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = obj.py();
        REPEAT
            .import(py, ITERTOOLS, "repeat")
            .and_then(|func| match n {
                Some(n) => func.call1((obj, n)),
                None => func.call1((obj,)),
            })
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn step_by<'py>(
        iterator: &Bound<'py, PyIterator>,
        step: &Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        ISLICE
            .import(py, ITERTOOLS, "islice")?
            .call1((iterator, 0, PyNone::get(py), step))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
    #[inline(always)]
    pub fn take<'py>(
        iterator: &Bound<'py, PyIterator>,
        stop: &Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterator.py();
        ISLICE
            .import(py, ITERTOOLS, "islice")?
            .call1((iterator, stop))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
}
/// Python `functools` module functions and objects
pub mod functools {
    use super::*;
    const FUNCTOOLS: &str = "functools";
    const REDUCE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    #[inline(always)]
    pub fn reduce<'py>(
        function: &Bound<'py, PyAny>,
        iterable: &Bound<'py, PyIterator>,
        initial: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = function.py();
        let args = match initial {
            Some(initial) => PyTuple::new(py, &[function, iterable, initial])?,
            None => PyTuple::new(py, &[function, iterable])?,
        };
        REDUCE.import(py, FUNCTOOLS, "reduce")?.call1(args)
    }
}
/// pyochain Python objects. This should not exist at the end of the migration.
pub mod pyochain {
    use super::*;
    const PYOCHAIN: &str = "pyochain";
    pub mod peekable {
        use super::*;
        const PEEKABLE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        #[inline(always)]
        pub fn new<'py>(iterable: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyIterator>> {
            let py = iterable.py();
            PEEKABLE
                .import(py, PYOCHAIN, "_peekable")?
                .getattr(intern!(py, "Peekable"))?
                .call1((iterable,))
                .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
        }
    }
    pub mod vec {
        use super::*;
        const VEC: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        #[inline(always)]
        pub fn from_ref<'py>(obj: &Bound<'py, PyList>) -> PyResult<Bound<'py, PySequence>> {
            let py = obj.py();
            VEC.import(py, PYOCHAIN, "Vec")?
                .getattr(intern!(py, "from_ref"))?
                .call1((obj,))
                .map(|obj| unsafe { obj.cast_into_unchecked::<PySequence>() })
        }
    }
}
