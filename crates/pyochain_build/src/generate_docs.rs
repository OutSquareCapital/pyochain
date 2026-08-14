use crate::paths::Root;
use crate::write;
use std::{collections::HashSet, fs, path::Path};
use tap::Pipe;
use toml::{Table, Value};

use crate::parse::PyClass;
use tap::prelude::*;

pub(super) fn run(root: &Root, docs: Root, pyclasses: &[PyClass]) -> Result<&'static str, String> {
    let docs_dir = docs
        .join("reference")
        .tap(|docs_ref| fs::create_dir_all(docs_ref).unwrap());
    let generated = pyclasses
        .iter()
        .map(|pyclass| handle_class(&docs_dir, pyclass))
        .collect::<HashSet<_>>();
    let pre_existing = docs
        .iter_on_extension("md")
        .map(|path| {
            docs.make_relative(&path)
                .0
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect::<HashSet<_>>();
    check_nav(root, generated, pre_existing)
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
fn check_nav(
    root: &Root,
    generated: HashSet<String>,
    pre_existing: HashSet<String>,
) -> Result<&'static str, String> {
    let mut nav_paths = HashSet::new();
    get_project(&root)
        .pipe_ref(get_references)
        .pipe(|references| collect_nav_paths(references, &mut nav_paths));
    let missing_paths = generated.difference(&nav_paths).pipe(join_paths);
    let invalid_nav_paths = nav_paths.difference(&pre_existing).pipe(join_paths);
    match (missing_paths.is_empty(), invalid_nav_paths.is_empty()) {
        (false, _) => Err(format!(
            "⚠️  Missing paths in navigation:\n {missing_paths}"
        )),
        (_, false) => Err(format!(
            "⚠️  Invalid paths in navigation:\n {invalid_nav_paths}"
        )),
        (true, true) => Ok("✓ Navigation is complete!"),
    }
}
fn get_project(root: &Root) -> toml::value::Table {
    let project = root
        .join("zensical.toml")
        .pipe(fs::read_to_string)
        .expect("Failed to read zensical.toml")
        .parse::<toml::Table>()
        .expect("Failed to parse zensical.toml")
        .remove("project")
        .expect("Failed to get `project` key from parsed file");
    match project {
        Value::Table(table) => table,
        _ => panic!("Invalid zensical.toml: 'project' should be a table"),
    }
}
fn get_references(project: &Table) -> &Value {
    project
        .get("nav")
        .expect("Failed to get nav")
        .as_array()
        .expect("Failed to get nav as array")
        .iter()
        .find_map(|item| item.get("API reference"))
        .expect("Failed to get API reference")
}
fn collect_nav_paths(value: &Value, paths: &mut HashSet<String>) {
    match value {
        Value::Table(table) => table
            .values()
            .for_each(|value| collect_nav_paths(value, paths)),
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_nav_paths(value, paths)),
        Value::String(path) => {
            paths.insert(path.clone());
        }
        _ => {}
    }
}

fn join_paths<'a>(paths: impl Iterator<Item = &'a String>) -> String {
    paths.map(String::as_str).collect::<Vec<_>>().join("\n")
}
