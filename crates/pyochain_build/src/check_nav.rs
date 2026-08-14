use crate::paths::Root;
use owo_colors::OwoColorize;
use std::{collections::HashSet, fs, path::Path};
use tap::Pipe;
use toml::{Table, Value};

pub(super) fn run(root: &Root) {
    anstream::eprintln!("{}", "Checking navigation completeness...".cyan().bold());
    let project = get_project(root);
    let docs_dirs = project["docs_dir"]
        .as_str()
        .expect("Failed to get `docs_dir` from zensical.toml");
    check_all(&project, &root.join(docs_dirs));
}

fn check_all(project: &Table, docs_dir: &Path) {
    let mut nav_paths = HashSet::new();
    get_references(project).pipe(|references| collect_nav_paths(references, &mut nav_paths));
    let missing_paths = missing_paths(&docs_dir, &nav_paths);
    let invalid_nav_paths = invalid_nav_paths(&docs_dir, &nav_paths);

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

fn missing_paths(docs_dir: &Path, nav_paths: &HashSet<String>) -> String {
    docs_dir
        .join("reference")
        .pipe(fs::read_dir)
        .expect("Failed to read reference directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("md"))
        .map(|path| relative_path(&path, docs_dir))
        .collect::<HashSet<_>>()
        .difference(nav_paths)
        .pipe(join_paths)
}

fn invalid_nav_paths(docs_dir: &Path, nav_paths: &HashSet<String>) -> String {
    let docs_paths = Root::new(docs_dir.to_path_buf())
        .iter_on_extension("md")
        .map(|path| relative_path(&path, docs_dir))
        .collect::<HashSet<_>>();
    nav_paths.difference(&docs_paths).pipe(join_paths)
}

fn relative_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .expect("Failed to strip root prefix from path")
        .to_string_lossy()
        .replace('\\', "/")
}

fn join_paths<'a>(paths: impl Iterator<Item = &'a String>) -> String {
    paths.map(String::as_str).collect::<Vec<_>>().join("\n")
}

fn show_warning(message: impl std::fmt::Display) {
    anstream::eprintln!("{}", message.yellow());
}
