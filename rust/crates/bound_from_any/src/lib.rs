mod enum_cast;
mod py_abc;
mod try_cast;
mod types;
use proc_macro::TokenStream;
use syn::{DeriveInput, ExprMatch, ItemTrait, parse_macro_input};

///`BoundFromAny` is a derive macro that generates a `FromPyObject` implementation for enums containing `Bound<'py, T>` variants.\
/// It allows matching a Python object against multiple possible PyO3 types using `cast` or `cast_exact`.\
/// The generated extractor tries variants in declaration order.\
/// The benefit vs deriving `FromPyObject` is that the actual type check is much more efficient and lightweight, since the scope is more narrow, and specialized for `Bound<'py, T>`.\
/// On tests, with 3 match arms, 2 success and one PyErr, the output assembly is the same as a hand-written implementation and try_cast! macro calls.\
/// **Note**: The claims of efficiency are verified by ASM cross-check with Gemini and GPT, I'm not an expert on this, so take it with a grain of salt).
///## Example
///
///```rust
///#[derive(BoundFromAny)]
///enum Container<'py> {
///    #[cast]
///    List(Bound<'py, PyList>),
///    #[cast_exact]
///    Tuple(Bound<'py, PyTuple>),
///}
/// fn process(obj: Bound<'_, PyAny>) -> PyResult<()> {
///    match obj.extract::<Container>()? {
///        Container::List(list) => {
///            println!("got a list");
///        }
///        Container::Tuple(tuple) => {
///            println!("got a tuple");
///        }
///    }
///
///    Ok(())
///}
///```
#[proc_macro_derive(BoundFromAny, attributes(cast, cast_exact, extract))]
pub fn derive_bound_from_any(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    enum_cast::generate_from_input(input)
}

/// Allow to define traits that acts just like python ABCs for pyclass types, i.e enforcing a common interface and providing default implementations for some methods.\
/// and avoids duplicating code by generating corresponding `#[pymethods]` impl blocks.\
/// This is needed because Pyo3 `#[pymethods]` macro does not allow to be used on traits and traits impl blocks.\
/// ## Usage
/// Use `#[py_abc(<pyclass types>)]` on a trait definition.\
/// The attribute arguments are the concrete `#[pyclass]` types which implement the trait.\
/// Use `#[py_skip]` on Rust-only helper methods.\
/// PyO3 method attributes such as `#[new]` and `#[pyo3(signature = (...))]` are forwarded.\
/// A generated Rust method is named `py_<trait method>` while its Python name remains the original trait method name.
/// ## Example
/// ```rust
/// #[py_abc(HeapMin, HeapMax)]
/// trait HeapType: Sized + PyWrapper<PyList> {
///   #[new]
/// fn new(data: Bound<'_, PyList>) -> PyResult<PyClassInitializer<Self>>;
/// #[py_skip]
/// fn from_ref<'py>(py: Python<'py>, data: Bound<'_, PyList>) -> PyResult<Bound<'py, Self>> {
///   Self::new(data).and_then(|init| Bound::new(py, init))
/// }
/// #[pyo3(signature = (item))]
/// fn replace<'py>(&self, item: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>;
/// }
/// ```
///
#[proc_macro_attribute]
pub fn py_abc(attr: TokenStream, item: TokenStream) -> TokenStream {
    let types = parse_macro_input!(attr with py_abc::parse_types);
    let item = parse_macro_input!(item as ItemTrait);
    py_abc::generate(item, types)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Ergonomic "flat" match of one or more `Bound<'_, PyAny>`-like values against concrete pyo3 types, using [`Bound::cast`].\
/// This avoids awkward nested `match` statements, or verbose `if let` chains.\
/// Unfortunately, we still have to use `match` keyword systematically, otherwise rustfmt will completely skip the code inside the macro.
///
/// ## Formatting tips
/// To make rustfmt fully format the code block, simply delete the macro call, run rustfmt, then re-add the macro call.\
/// As such, prefer using `{ ... }` instead of `(...)` at call sites, to just have to delete `try_cast!`.
///
/// ## Example
/// ```rust
/// use pyo3::prelude::*;
/// use pyo3::types::{PyDict, PyList, PyTuple};
/// use pyo3_ext::try_cast;
/// use pyo3::exceptions::PyTypeError;
///
/// fn foo(value: Bound<'_, PyAny>) -> PyResult<isize> {
///   try_cast! {match value {
///     PyList | PyTuple => { Ok(0) }
///     PyDict => { Ok(1) }
///     _ => { Err(PyTypeError::new_err("Invalid type")) }
///    }}
///  }
/// fn foo_no_macro(value: Bound<'_, PyAny>) -> PyResult<()> {
///   match value.cast::<PyList>() {
///     Ok(_) => Ok(0),
///     Err(_) => match value.cast::<PyTuple>() {
///         Ok(_) => Ok(1),
///         Err(_) => match value.cast::<PyDict>() {
///             Ok(_) => Ok(2),
///             Err(_) => Err(PyTypeError::new_err("Invalid type")),
///         }
///     }
/// }
/// ```
#[proc_macro]
pub fn try_cast(input: TokenStream) -> TokenStream {
    let match_expr = parse_macro_input!(input as ExprMatch);
    try_cast::generate_from_expr(match_expr)
}
