mod enum_cast;
mod try_cast;
mod types;
use proc_macro::TokenStream;
use syn::{DeriveInput, ExprMatch, parse_macro_input};

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

/// Ergonomic "flat" match of one or more `Bound<'_, PyAny>`-like values against concrete pyo3 types, using [`Bound::cast`].\
/// This avoids awkward nested `match` statements, or verbose `if let` chains.\
/// ## Current limitations
/// - Unfortunately, we still have to use `match` keyword systematically, otherwise rustfmt will completely skip the code inside the macro.
/// - For some reason, rustfmt will delete `,` statements between the match arms, so we need to handle that specially without requiring them, which diverge from regular rust syntax.
///
/// ## Example
/// ```rust
/// use pyo3::prelude::*;
/// use pyo3::types::{PyDict, PyList, PyTuple};
/// use pyo3_ext::try_cast;
/// use pyo3::exceptions::PyTypeError;
///
/// fn foo(value: Bound<'_, PyAny>) -> PyResult<isize> {
///   try_cast!(match value {
///     PyList | PyTuple => { Ok(0) }
///     PyDict => { Ok(1) }
///     _ => { Err(PyTypeError::new_err("Invalid type")) }
///    })
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
