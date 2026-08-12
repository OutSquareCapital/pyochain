use crate::{
    abc, iterators,
    pyovec::PyoVec,
    traits::{PyWrapper, PyoABC},
};
use either::Either;
use pyo3::{
    PyTypeInfo, intern,
    prelude::*,
    types::{PyList, PyNotImplemented, PyTuple},
};
use pyo3_ext::{
    pylibs,
    types::{FromCmp, PyIterable},
};
use pyochain_macros::{BoundFromAny, py_abc, try_cast};
use tap::Pipe;
/// Enum used to convert various types into a `PyList` for heap operations.
#[derive(BoundFromAny)]
enum IntoHeap<'py> {
    #[cast_exact]
    List(Bound<'py, PyList>),
    #[cast_exact]
    Vec(Bound<'py, PyoVec>),
    Iterable(Bound<'py, PyIterable>),
}
impl IntoHeap<'_> {
    fn convert<F: FnOnce(&Bound<'_, PyList>) -> PyResult<()>>(
        self,
        func: F,
    ) -> PyResult<Py<PyList>> {
        match self {
            Self::List(list) => {
                func(&list)?;
                Ok(list)
            }
            Self::Vec(vec) => {
                let py = vec.py();
                let inner = vec.get().into_inner_bound(py);
                func(&inner)?;
                Ok(inner)
            }
            Self::Iterable(iterable) => {
                let py = iterable.py();
                let list = PyList::type_object(py)
                    .call1((iterable,))
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;
                func(&list)?;
                Ok(list)
            }
        }
        .map(Bound::unbind)
    }
}
#[py_abc(HeapMin, HeapMax)]
trait HeapType: Sized + PyWrapper<PyList> {
    #[new]
    fn new(data: IntoHeap<'_>) -> PyResult<PyClassInitializer<Self>>;

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name = Self::type_object(py).name()?;
        let repr = self.inner_bind(py).repr()?.to_string();
        Ok(format!("{}({})", name, repr))
    }
    fn replace<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>;
    fn push_pop<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>;
    fn push<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<()>;

    #[pyo3(signature = (_index=None))]
    fn pop<'py>(
        &self,
        py: Python<'py>,
        _index: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>>;
    #[staticmethod]
    fn from_ref<'py>(py: Python<'py>, data: Bound<'_, PyList>) -> PyResult<Bound<'py, Self>>;
    fn __len__(&self, py: Python<'_>) -> usize {
        self.inner_bind(py).len()
    }

    fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.inner_bind(index.py()).as_any().get_item(index)
    }

    fn __setitem__(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(index.py()).as_any().set_item(index, value)
    }

    fn __delitem__(&self, index: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(index.py()).as_any().del_item(index)
    }

    fn __eq__<'py>(
        &self,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Either<bool, Bound<'py, PyNotImplemented>>> {
        let py = other.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match other {
                CaseExact::HeapMax(x) | CaseExact::HeapMin(x) | CaseExact::PyoVec(x) => inner.eq(x.get().inner().clone_ref(py)).map(Either::Left),
                Case::PyList(list) => inner.eq(list).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn insert(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py())
            .call_method1("insert", (index, value))?;
        Ok(())
    }

    #[pyo3(signature = (*others, key=None, reverse=false))]
    fn merge<'py>(
        &self,
        others: Bound<'py, PyTuple>,
        key: Option<Bound<'py, PyAny>>,
        reverse: bool,
    ) -> PyResult<Bound<'py, iterators::Iter>> {
        let py = others.py();
        let args = self
            .into_inner_bound(py)
            .into_any()
            .pipe(std::iter::once)
            .chain(others.iter())
            .collect::<Vec<_>>()
            .pipe(|x| PyTuple::new(py, x))?;
        pylibs::heapq::merge(py, args, key, reverse).and_then(iterators::Iter::new)
    }
    #[pyo3(signature = (n, key=None))]
    fn n_smallest<'py>(
        &self,
        py: Python<'py>,
        n: isize,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        pylibs::heapq::nsmallest(n, self.inner_bind(py).as_any(), key)
            .and_then(|x| Self::from_ref(py, x))
    }
    #[pyo3(signature = (n, key=None))]
    fn n_largest<'py>(
        &self,
        py: Python<'py>,
        n: isize,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        pylibs::heapq::nlargest(n, self.inner_bind(py).as_any(), key)
            .and_then(|x| Self::from_ref(py, x))
    }
}
/// Present for typing purposes only.
#[pyclass(module = "pyochain.collections",frozen, generic, sequence, extends = abc::PyoMutableSequence, subclass)]
pub struct Heap;
#[pyclass(module = "pyochain.collections",frozen, generic, sequence, extends = Heap)]
pub struct HeapMin(pub Py<PyList>);
#[pymethods]
impl Heap {
    #[allow(unused_variables)]
    #[new]
    fn new(data: Bound<'_, PyList>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
}
impl HeapType for HeapMin {
    fn new(data: IntoHeap<'_>) -> PyResult<PyClassInitializer<Self>> {
        data.convert(pylibs::heapq::heapify)
            .map(|inner| Heap::build_init().add_subclass(Self(inner)))
    }
    fn from_ref<'py>(py: Python<'py>, data: Bound<'_, PyList>) -> PyResult<Bound<'py, Self>> {
        let initializer = Heap::build_init().add_subclass(Self(data.unbind()));
        Bound::new(py, initializer)
    }
    fn push(&self, item: Bound<'_, PyAny>) -> PyResult<()> {
        pylibs::heapq::heappush(self.inner_bind(item.py()), item)
    }
    fn pop<'py>(
        &self,
        py: Python<'py>,
        _index: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        pylibs::heapq::heappop(self.inner_bind(py))
    }
    fn replace<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        pylibs::heapq::heapreplace(self.inner_bind(item.py()), item)
    }

    fn push_pop<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        pylibs::heapq::heappushpop(self.inner_bind(item.py()), item)
    }
}

#[pyclass(module = "pyochain.collections",frozen, generic, sequence, extends = Heap)]
pub struct HeapMax(pub Py<PyList>);
impl HeapType for HeapMax {
    fn new(data: IntoHeap<'_>) -> PyResult<PyClassInitializer<Self>> {
        data.convert(pylibs::heapq::heapify_max)
            .map(|inner| Heap::build_init().add_subclass(Self(inner)))
    }
    fn from_ref<'py>(py: Python<'py>, data: Bound<'_, PyList>) -> PyResult<Bound<'py, Self>> {
        let initializer = Heap::build_init().add_subclass(Self(data.unbind()));
        Bound::new(py, initializer)
    }
    fn push(&self, item: Bound<'_, PyAny>) -> PyResult<()> {
        let py = item.py();
        let inner = self.inner_bind(py);
        inner.append(item)?;
        self._siftdown(py, 0, inner.len() - 1)
    }
    fn pop<'py>(
        &self,
        py: Python<'py>,
        _index: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner_bind(py);
        let lastelt = inner.call_method0(intern!(py, "pop"))?;
        if !(inner.is_empty()) {
            let returnitem = inner.get_item(0)?;
            inner.set_item(0, lastelt)?;
            self._siftup(py, 0)?;
            Ok(returnitem)
        } else {
            Ok(lastelt)
        }
    }

    fn replace<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = item.py();
        let inner = self.inner_bind(py);
        let returnitem = inner.get_item(0)?; // raises appropriate IndexError if heap is empty
        inner.set_item(0, item)?;
        self._siftup(py, 0)?;
        Ok(returnitem)
    }
    fn push_pop<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = item.py();
        let inner = self.inner_bind(py);
        if !(inner.is_empty()) && item.lt(inner.get_item(0)?)? {
            let returnitem = inner.get_item(0)?;
            inner.set_item(0, item)?;
            self._siftup(py, 0)?;
            Ok(returnitem)
        } else {
            Ok(item)
        }
    }
}
impl HeapMax {
    fn _siftdown(&self, py: Python<'_>, startpos: usize, mut pos: usize) -> PyResult<()> {
        let inner = self.inner_bind(py);
        let newitem = inner.get_item(pos)?;
        // Follow the path to the root, moving parents down until finding a place
        // newitem fits.
        while pos > startpos {
            let parentpos = (pos - 1) >> 1;
            let parent = inner.get_item(parentpos)?;
            if parent.lt(&newitem)? {
                inner.set_item(pos, parent)?;
                pos = parentpos;
                continue;
            }
            break;
        }
        inner.set_item(pos, newitem)
    }

    fn _siftup(&self, py: Python<'_>, mut pos: usize) -> PyResult<()> {
        let inner = self.inner_bind(py);
        let endpos = inner.len();
        let startpos = pos;
        let newitem = inner.get_item(pos)?;
        // Bubble up the larger child until hitting a leaf.
        let mut childpos = 2 * pos + 1; // leftmost child position
        while childpos < endpos {
            // Set childpos to index of larger child.
            let rightpos = childpos + 1;
            if rightpos < endpos && !(inner.get_item(rightpos)?.lt(inner.get_item(childpos)?)?) {
                childpos = rightpos;
            }
            // Move the larger child up.
            inner.set_item(pos, inner.get_item(childpos)?)?;
            pos = childpos;
            childpos = 2 * pos + 1;
        }
        // The leaf at pos is empty now.  Put newitem there, and bubble it up
        // to its final resting place (by sifting its parents down).
        inner.set_item(pos, newitem)?;
        self._siftdown(py, startpos, pos)
    }
}
