# Possibilities of implementations for Option and Result types in Python

This document outlines three different approaches to implementing `Option` and `Result` types in Python, discussing their pros and cons.

## Differences between the 2 types

- `Option` has only one generic type parameter (the contained value type), while `Result` has two (the success value type and the error value type).
- `Result` can easily be instanciated directly from it's concrete classes (`Ok` and `Err`), while `Option` needs a factory function to create the correct variant (`Some` or `NoneOption`, which is a singleton).
-> This could be changed. Singleton have been chosen for `NoneOption` to save memory, but the concrete gain has not been measured. However the Any type of `NoneOption` is sometimes problematic for type inference.
- Pattern matching works out of the box with separate classes, but won't with a common base class, period. Only type union will work.
- Using Self VS the abstract base class/type alias as return types is not something that is really determined as better for one or the other. The type inference will be the main factor here.
- Priority are in this order: typing friendliness (match pattern, type inference, etc.) > User API convenience > performance > code duplication.

This mean that some approaches that work well for `Option` may not be as suitable for `Result`, and vice versa.

## 1. Separate classes, type alias

[see the file here](type_approaches\_first.py)
**Pros**:

- Best performance by far (no runtime checks)
- Pattern matching works

**Cons**:

- MASSIVE code duplication
- Must have `option` function for creation (rather than `Option.from_`) -> duplicate type and function names - Can't use `Option` for instance checks
- NoneOption must stay generic.
- No true common interface, loss of Protocol benefits

## 2. Common abstract base class + type alias

[see the file here](type_approaches\_second.py)

**Pros**:

- No code duplication
- Common interface
- Pattern matching works
- NoneOption don't care about genericity

**Cons**:

- Runtime checks for every method call (is_err, unwrap, etc) -> performance hit
- Must have `option` function for creation (rather than `Option.from_`) -> duplicate type and function names - Can't use `Option` for instance checks

## 3. Common abstract base class + generic subclasses (currently implemented)

**Pros**:

- No code duplication
- Simplest internal implementation
- Option.from_ pattern -> can be extended further
- The base class IS the type -> Can be used for signatures and instance checks
- NoneOption don't care about genericity

**Cons**:

- Runtime checks for every method call (is_err, unwrap, etc) -> performance hit
- Pattern matching don't work (will always expect base class)

## 4. Common abstract base class as pure interface + pure implementations subclasses

[see the file here](type_approaches\_third.py) #! TODO
**Pros**:

- No code duplication
- Common interface
- Performance as good as first approach (no runtime checks)
- Compared to first solution, don't repeat the docstrings (which are the bulk of the LOC)
- Option.from_ pattern -> can be extended further
- The base class IS the type -> Can be used for signatures and instance checks
**Cons**:
- Pattern matching don't work (will always expect base class)
- More complex internal implementation
