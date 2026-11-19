//! `pyo3` doesn't expose two small `CPython` functions that let us read small integers quickly
//! (`PyUnstable_Long_IsCompact` and `PyUnstable_Long_CompactValue`), so `ffi.rs` declares them itself.
//!
//! On Linux and macOS that's all it takes: the linker leaves them blank and Python fills them in
//! when the module is imported. Windows doesn't work that way. There, the linker has to be told
//! which DLL each function comes from, and that DLL is named after the Python version
//! (`python313.dll`, `python314.dll`, ...). `pyo3` does this for its own functions, but keeps the
//! machinery private, so we can't reuse it.
//!
//! So this script asks `pyo3` which interpreter we're building for and writes the declarations
//! with the right DLL name attached. The DLL name only matters on Windows.
//! NOTE: Wrote by Claude. Before that it wasn't compiling on my machine.

use std::{env, fs, path::PathBuf};

fn main() {
    let dll = pyo3_build_config::get().lib_name().unwrap_or("python3");
    let text = format!(
        r#"#[cfg_attr(windows, link(name = "{dll}", kind = "raw-dylib"))]
unsafe extern "C" {{
    pub fn PyUnstable_Long_IsCompact(op: *const pyo3::ffi::PyLongObject) -> std::ffi::c_int;
    pub fn PyUnstable_Long_CompactValue(op: *const pyo3::ffi::PyLongObject) -> pyo3::ffi::Py_ssize_t;
}}
"#
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("python_ffi.rs");
    fs::write(out, text).expect("failed to write python_ffi.rs");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
}
