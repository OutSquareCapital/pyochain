# Pyochain architecture and conventions

The following documentation provides an overview of the project architecture and code conventions.

It's destined to both humans and LLMs, and the fact that it's in `AGENTS.md` is merely a convenience to enforce it in the local context when working with agents.

## Architecture

The root crate assembles the public extension module.

`src/lib.rs` is the single registration point for the python package, and also for functionality that need to be run at import time.

### Internal Crates

We use several internal crates for faster compile times and better modularity.

If you want to find/edit/add some code, match your task to it. Does the code:

- extend `pyo3` with traits extensions, new standard-library helpers/types, ffi, etc? => [`pyo3_ext`](crates\pyo3_ext)
- is a proc macro? => [`pyochain_macros`](crates\pyochain_macros)
- extends rust std? => [`std_tools`](crates\std_tools)
- is directly related to sorted collections internal implementations? => [`sorted_rs`](crates\sorted_rs)
- is related to pyochain's public API or high-level behavior? => [`src`](src)

## Code Guidelines

### Style

#### General

The guidelines below apply to both rust source code and python scripts.

1. We use a descending order for organizing code: types, then public symbols, then private helpers.
2. Prefer chained method calls over intermediate variables. As such, generous use of `tap::{pipe, tap}` crate across the code is encouraged.
3. It should come with no surprise, given the library purpose, that we don't like imperative style `for i in iterable if condition` and vastly prefer `iterable.into_iter().filter(...).map(...)`.
4. We generally avoid inline comments (`//`, `#`). They should only be used for local implementation details, or `TODO:`/`NOTE:`/`BUG` annotations.
5. Documentation comments (`///`, `"""..."""`) are mandatory in python stubs, but outside of that should be used judiciously. They should focus on intended usage, behavior and eventual tips, rather than implementation details.
6. Functionality locality matters: free functions should be avoided if it can clearly be linked to an appropriate type (as associated function/method, classmethod/staticmethod, etc...).
7. Prefer `if condition {...} else {...}` over early returns.

#### Rust-specifics

- macros are powerful, macros are cool, but LSP and formatters don't understand them well. If you can, extract as much as possible out of it. For example, `fn foo{...}` can become `fn foo{helper_function(...)}`. Apply this with good sense however, and avoid over-engineering trivial cases.
- The function locality with traits has a special case: since a pub trait imply all methods are public, free functions can be used more liberally to keep public interfaces clean and concise.

### Implementations

#### local abstractions

Use the existing abstractions before adding a local equivalent.

- `py_abc` is one of the main abstractions for defining Python-visible behavior in Rust. Consult related documentation for proper usage.
- Raw accesses like `obj.callmethod(...)` are almost never needed. If you find yourself using it, consider implementing it in `pyo3_ext` instead.

#### Pyo3

- Prefer `self, py: Python<...>` as the default signature for methods that need Python context. `slf: Bound<'_, Self>` can be used when necessary.
- A pyclass should be frozen by default. If not, it must be explictly motivated by a clear reason.

#### python -> rust conversions

There's a LOT of different ways to convert a ``Bound<`_, PyAny>`` to a Rust type. match the situation to choose the most appropriate method.

##### Multiple conversions

| Converter                         | to                                                        | Pros                                                                                                      | Cons                                                                                                                   | Use when                                                                                                                                           | Tips                                                                                                                                                         |
| --------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `try_cast!`                       | match branches`{Bound<'_, &T>, Bound<'_, &U>, ...}`       | Convenient, fast, readable, cast OR cast_exact, no need to invent types and name them                     | macro formatter pitfalls, can only `Bound<'_, T>` -> `Bound<'_, &U>` for the python specifics                          | Default for many conversions, `PyNotImplemented` on failure                                                                                        | Still work as a regular rust match so can handle `PyAny` nested inside rust enums, who may have variants that are not even `IntoPyObject` or boolean guards. |
| `try_cast_into!`                  | match branches `{Bound<'_, T>, Bound<'_, U>, ...}`        | Same as `try_cast!` but owned                                                                             | .                                                                                                                      | Fallback if `try_cast!` don't satisfy borrowing requirements                                                                                       | .                                                                                                                                                            |
| `#derive[BoundFromAny]`           | enum variants `Foo(Bound<'_, T>), Bar(Bound<'_, T>), ...` | Reusable, abstracts conversion logic, can `cast`, `cast_exact` and `extract`, usable in python signatures | Heavier to implement and use, bit less efficient with the enum wrapper, clunkier to gracefully handle conversion error | `Either` is not enough, mixed `extract` and `cast` paths, must live trough multiple layers of functions calls, `PyTypeError` on conversion failure | For custom handling of a failed conversion, include a variant for `PyAny`.                                                                                   |
| `either::Either<T, U>`            | `Either::Left(T), Either::Right(U)`                       | Convenient for handling two possible types, reusable, usable in python signatures                         | Called from python, limited to two types, implicit error early return                                                  | When you only need to handle exactly two possible types, `PyTypeError` on failure                                                                  | Define a type alias for readability if used frequently.                                                                                                      |
| enum with `#derive[FromPyObject]` | enums variants                                            | All solutions below are built on top of this trait, full pyo3 integration, usable in python signatures    | Default implementation is much slower than alternatives, more complex to implement                                     | No alternative is suitable                                                                                                                         | .                                                                                                                                                            |

##### Single conversions

| Converter                              | to                                 | Pros                             | Cons                                                                 | Use when                                                                                                                                      | Tips                                                                |
| -------------------------------------- | ---------------------------------- | -------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `Bound::cast_unchecked::<T>(...)`      | `Bound<'_, &T>`                    | Fastest conversion               | unsafe                                                               | You can statically guarantee the type, e.g `PyList.try_iter().unwrap()` WILL return a `PyIterator`. Work for subclassing as well.             | `iter_py` can do the `PyIterator` conversion for that precise case. |
| `Bound::cast_into_unchecked::<T>(...)` | `Bound<'_, T>`                     | .                                | .                                                                    | `cast_unchecked` can't satisfy borrowing requirements, or you want an unsafe `extract`.                                                       | .                                                                   |
| `Bound::cast_exact::<T>(...)`          | `Result<Bound<'_, &T>, CastError>` | Fastest **safe** conversion      | Won't handle subclassing, need to manually be converted to a `PyErr` | `T` is a pyochain type without `subclass` support, or you don't care about MRO resolution, , or a conversion failure as a non-error fallback. | .                                                                   |
| `Bound::cast::<T>(...)`                | `Result<Bound<'_, &T>, CastError>` | Safe and flexible conversion     | Slower than `cast_exact`, need to manually be converted to a `PyErr` | You want to handle subclassing, or a conversion failure as a non-error fallback                                                               | .                                                                   |
| `Bound::cast_into::<T>(...)`           | `Result<Bound<'_, T>, CastError>`  |
| `Bound::cast_into_exact::<T>(...)`     | `Result<Bound<'_, T>, CastError>`  |
| `Bound::extract::<T>(...)`             | `Result<T, CastError>`             | Directly extract to Rust structs | Slower than `cast`, need to manually be converted to a `PyErr`       | You need to convert Python to rust struct that aren't pyclasses, e.g `usize`, `String`, or custom Rust types.                                 | Use `Bound::{cast, cast_into}.get()` instead on PyClass instances.  |
