use crate::paths::Root;
use crate::write;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::parse::PyClass;
use tap::prelude::*;

pub(super) struct DocsPaths {
    pub(super) generated: HashSet<String>,
    pub(super) all: HashSet<String>,
}

pub(super) fn run(docs: Root, pyclasses: &[PyClass]) -> DocsPaths {
    let docs_dir = docs
        .join("reference")
        .tap(|docs_ref| fs::create_dir_all(docs_ref).unwrap());
    let generated = pyclasses
        .iter()
        .map(|pyclass| handle_class(&docs_dir, pyclass))
        .collect();
    let all = docs
        .iter_on_extension("md")
        .map(|path| {
            docs.make_relative(&path)
                .0
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    DocsPaths { generated, all }
}
#[inline]
fn handle_class(docs_ref: &Path, pyclass: &PyClass) -> String {
    let name = &pyclass.python_name;
    let filename = format!("{}.md", name.to_lowercase());
    let path = docs_ref.join(&filename);
    let full_path = format!("{}.{}", pyclass.module.as_ref().unwrap(), name);
    let new_content = write::get_new_content(name, &full_path);
    write::Kind::new(&path, &new_content).maybe_write(&path, new_content);
    format!("reference/{filename}")
}
