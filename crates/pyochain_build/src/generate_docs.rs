use crate::{
    parse::{PyClass, RegisteredClassVisitor},
    paths,
};
use owo_colors::OwoColorize;
use std::{collections::HashSet, fs, path::PathBuf, process};

use tap::prelude::*;

pub(super) fn run(
    root: paths::Related,
    src: paths::Related,
    docs: paths::Related,
    stubs: paths::Related,
) -> Result<(), Box<dyn std::error::Error>> {
    let pyclasses = src
        .child
        .pipe_ref(fs::read_to_string)
        .unwrap()
        .pipe(RegisteredClassVisitor::visit)
        .pipe(|registered_classes| {
            src.iter_on_extension("rs")
                .flat_map(|path: PathBuf| {
                    path.pipe_ref(fs::read_to_string)
                        .expect("Failed to read source file")
                        .pipe_ref(|source| syn::parse_file(source))
                        .expect("Failed to parse source file")
                        .items
                        .into_iter()
                        .filter_map(|item| PyClass::from_item(item, src.make_relative(&path)))
                        .filter(|pyclass| registered_classes.contains(&pyclass.rust_name))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
    let nav_paths = root
        .child
        .pipe_ref(fs::read_to_string)?
        .parse::<toml::Table>()?
        .pipe(get_nav_paths);
    let known_paths = docs
        .iter_on_extension("md")
        .map(|path| {
            docs.make_relative(&path)
                .0
                .to_string_lossy()
                .replace('\\', "/")
        })
        .chain(pyclasses.iter().map(PyClass::document_path))
        .collect::<HashSet<_>>();
    let error_count = pyclasses
        .iter()
        .map(|pyclass| pyclass.check(&stubs, &docs.child, &nav_paths))
        .chain(nav_paths.difference(&known_paths).map(|path| {
            Err(format!(
                "invalid path in navigation\nNavigation path: {path}"
            ))
        }))
        .enumerate()
        .try_fold(0, |error_count, (index, result)| match result {
            Ok((path, content)) => {
                anstream::eprintln!("{}", write_document(&root, path, content)?.cyan().bold());
                Ok::<usize, Box<dyn std::error::Error>>(error_count)
            }
            Err(message) => {
                anstream::eprintln!("\n{:>3}. {}", index + 1, message.red().bold());
                Ok(error_count + 1)
            }
        })?;
    finalize(error_count, pyclasses.len());
    Ok(())
}
fn finalize(error_count: usize, pyclasses_nb: usize) {
    if error_count == 0 {
        println!("{}", "✓ Navigation is complete!".green().bold());
    } else {
        anstream::eprintln!(
            "{}: {} issue(s) found in {} pyclass declaration(s).",
            "Documentation generation failed".red().bold(),
            error_count.red().bold(),
            pyclasses_nb.cyan(),
        );
        anstream::eprintln!("============================================================");
        process::exit(1);
    }
}

fn write_document(
    root: &paths::Related,
    path: PathBuf,
    content: String,
) -> Result<String, std::io::Error> {
    let display_path = root.make_relative(&path).0.display();
    let result = match path.exists() {
        false => {
            fs::write(&path, content)?;
            format!("Generated {display_path} (new file)")
        }
        true if fs::read_to_string(&path)? == content => {
            format!("Skipping {display_path} (no changes)")
        }
        true => {
            fs::write(&path, content)?;
            format!("Updating {display_path} (co§ntent changed)")
        }
    };
    Ok(result)
}

fn get_nav_paths(parsed: toml::map::Map<String, toml::Value>) -> HashSet<String> {
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
    let mut paths = HashSet::new();
    collect_nav_paths(api_ref, &mut paths);
    paths
}
fn collect_nav_paths(value: &toml::Value, paths: &mut HashSet<String>) {
    match value {
        toml::Value::Table(table) => table
            .values()
            .for_each(|value| collect_nav_paths(value, paths)),
        toml::Value::Array(values) => values
            .iter()
            .for_each(|value| collect_nav_paths(value, paths)),
        toml::Value::String(path) => {
            paths.insert(path.clone());
        }
        _ => {}
    }
}
