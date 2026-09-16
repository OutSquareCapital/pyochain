//! This module contains Python built-in functions and objects, as well as functions and objects from the `itertools` and `functools` modules.\
//! Each submodule declares a const string with the name of the module, and a const `PyOnceLock` + associated fn for each function or object that is imported from that module.\
//! This pattern ensure maximum performance by only importing the function or object once, and reusing it for subsequent calls.
//! We also use unsafe casts to correct types, aggressive inlining, and `&Bound` to maximize performance.
pub mod builtins;
pub mod functools;
pub mod heapq;
pub mod itertools;
pub mod operator;
pub mod sys;
