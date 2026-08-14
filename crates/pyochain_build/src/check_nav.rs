use crate::generate_docs::DocsPaths;
use crate::paths::Root;
use owo_colors::OwoColorize;
use std::{collections::HashSet, fs};
use tap::Pipe;
use toml::{Table, Value};

pub(super) fn run(root: Root, docs_paths: DocsPaths) {
    anstream::eprintln!("{}", "Checking navigation completeness...".cyan().bold());
    let project = get_project(&root);
    let mut nav_paths = HashSet::new();
    get_references(&project).pipe(|references| collect_nav_paths(references, &mut nav_paths));
    let missing_paths = docs_paths.generated.difference(&nav_paths).pipe(join_paths);
    let invalid_nav_paths = nav_paths.difference(&docs_paths.all).pipe(join_paths);

    if !missing_paths.is_empty() {
        show_warning(format_args!(
            "⚠️  Missing generated files in zensical.toml:\n {missing_paths}"
        ));
    }
    if !invalid_nav_paths.is_empty() {
        show_warning(format_args!("⚠️  Invalid nav links:\n {invalid_nav_paths}"));
    }

    if missing_paths.is_empty() && invalid_nav_paths.is_empty() {
        anstream::eprintln!("{}", "✓ Navigation is complete!".green());
    } else {
        anstream::eprintln!(
            "{}",
            "❌ Please fix the above issues before deploying the documentation."
                .red()
                .bold()
        );
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

fn show_warning(message: impl std::fmt::Display) {
    anstream::eprintln!("{}", message.yellow());
}
