use crate::{
    parse::{PyClass, RegisteredClassVisitor},
    paths,
};
use owo_colors::OwoColorize;
use std::{collections::HashSet, error::Error, fs, path::PathBuf};

use tap::prelude::*;

pub(super) fn run(
    root: paths::Related,
    src: paths::Related,
    docs: paths::Related,
    stubs: paths::Related,
) -> Result<(), Box<dyn Error>> {
    let registered_classes = src
        .child
        .pipe_ref(fs::read_to_string)
        .unwrap()
        .pipe(RegisteredClassVisitor::visit);
    let nav_path_checks = root
        .child
        .pipe_ref(fs::read_to_string)?
        .parse::<toml::Table>()?
        .pipe(get_nav_paths)
        .into_iter()
        .filter_map(|path| {
            if docs
                .child
                .join(path.trim_start_matches("reference/"))
                .is_file()
            {
                None
            } else {
                let msg = format!("invalid path in navigation\nNavigation path: {path}");
                Some(Err(msg))
            }
        });

    src.iter()
        .flat_map(|path| extract_classes_from_file(path, &src, &registered_classes))
        .map(|pyclass| pyclass.check(&stubs, &docs.child))
        .chain(nav_path_checks)
        .try_fold(0, |error_count, result| match result {
            Ok((path, content)) => {
                anstream::eprintln!("{}", root.write(path, content)?.cyan().bold());
                Ok::<usize, Box<dyn std::error::Error>>(error_count)
            }
            Err(message) => {
                anstream::eprintln!("{}", message.red().bold());
                Ok(error_count + 1)
            }
        })
        .map(finalize)
}

fn finalize(error_count: usize) {
    if error_count == 0 {
        println!("{}", "✓ Navigation is complete!".green().bold());
    } else {
        anstream::eprintln!(
            "\n{}: {} issue(s) found.",
            "Documentation generation failed".red().bold(),
            error_count.red().bold(),
        );
        anstream::eprintln!("============================================================");
    }
}
fn extract_classes_from_file(
    path: PathBuf,
    src: &paths::Related,
    registered_classes: &HashSet<String>,
) -> Vec<PyClass> {
    path.pipe_ref(fs::read_to_string)
        .expect("Failed to read source file")
        .pipe_ref(|source| syn::parse_file(source))
        .expect("Failed to parse source file")
        .items
        .into_iter()
        .filter_map(|item| PyClass::from_item(item, src.make_relative(&path)))
        .filter(|pyclass| registered_classes.contains(&pyclass.rust_name))
        .collect::<Vec<_>>()
}
fn get_nav_paths(parsed: toml::map::Map<String, toml::Value>) -> Vec<String> {
    let api_ref = parsed
        .get("project")
        .expect("Failed to get `project` key from parsed file")
        .as_table()
        .expect("Invalid zensical.toml: 'project' should be a table")
        .get("nav")
        .expect("Failed to get nav")
        .as_array()
        .expect("Failed to get nav as array")
        .iter()
        .find_map(|item| item.get("API reference"))
        .expect("Failed to get API reference");
    let mut paths = Vec::new();
    collect_nav_paths(api_ref, &mut paths);
    paths
}
fn collect_nav_paths(value: &toml::Value, paths: &mut Vec<String>) {
    match value {
        toml::Value::Table(table) => table
            .values()
            .for_each(|value| collect_nav_paths(value, paths)),
        toml::Value::Array(values) => values
            .iter()
            .for_each(|value| collect_nav_paths(value, paths)),
        toml::Value::String(path) => {
            paths.push(path.clone());
        }
        _ => {}
    }
}
