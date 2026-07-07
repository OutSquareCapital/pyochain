use std::{fmt, fs, path::Path};

/// A value found while scanning source, tagged with where it came from so a failing check
/// can point directly at the file (and line, for Rust source) to fix.
struct Found {
    value: String,
    location: String,
}

impl fmt::Debug for Found {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} ({})", self.value, self.location)
    }
}

fn main() {
    let env = env!("CARGO_MANIFEST_DIR");
    let pyochain_dir = Path::new(env).join("../src/pyochain");
    let src_dir = Path::new(env).join("src");
    println!("cargo:rerun-if-changed={}", pyochain_dir.display());

    let (expected_names, expected_modules) = expected(&pyochain_dir);
    let (declared_names, declared_modules) = declared(&src_dir);

    assert_subset(
        &expected_names,
        &declared_names,
        "missing `#[pymodule(name = \"...\")]` declaration(s) in rust/src for stub file(s)",
    );
    assert_subset(
        &declared_modules,
        &expected_modules,
        "#[pyclass(module = \"...\")] value(s) don't match any known stub path",
    );
    assert_subset(
        &expected_modules,
        &declared_modules,
        "stub path(s) with no `#[pyclass(module = \"...\")]` declaring them",
    );
}

fn declared(dir: &Path) -> (Vec<Found>, Vec<Found>) {
    fn extract<'a>(
        content: &'a str,
        path: &'a Path,
        marker: &'a str,
    ) -> impl Iterator<Item = Found> + 'a {
        content.match_indices(marker).filter_map(move |(i, m)| {
            let rest = &content[i + m.len()..];
            let line = content[..i].matches('\n').count() + 1;
            rest.find('"').map(|end| Found {
                value: rest[..end].to_owned(),
                location: format!("{}:{line}", path.display()),
            })
        })
    }
    dir.read_dir()
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .inspect(|path| println!("cargo:rerun-if-changed={}", path.display()))
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("rs"))
        .fold(
            (Vec::new(), Vec::new()),
            |(mut names, mut modules), path| {
                let content = fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
                names.extend(extract(&content, &path, "#[pymodule(name = \""));
                modules.extend(extract(&content, &path, "module = \""));
                (names, modules)
            },
        )
}
fn expected(dir: &Path) -> (Vec<Found>, Vec<Found>) {
    dir.read_dir()
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .inspect(|path| println!("cargo:rerun-if-changed={}", path.display()))
        .map(|path| {
            if path.is_dir() {
                on_dir(&path)
            } else if path.extension().and_then(|e| e.to_str()) == Some("pyi") {
                on_file(&path)
            } else {
                (Vec::new(), Vec::new())
            }
        })
        .fold(
            (Vec::new(), Vec::new()),
            |(mut names, mut paths), (n, p)| {
                names.extend(n);
                paths.extend(p);
                (names, paths)
            },
        )
}

fn on_dir(path: &Path) -> (Vec<Found>, Vec<Found>) {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|folder| {
            let (file_names, paths): (Vec<Found>, Vec<Found>) = path
                .read_dir()
                .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .inspect(|path| println!("cargo:rerun-if-changed={}", path.display()))
                .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("pyi"))
                .filter_map(|file| {
                    file.file_stem().and_then(|s| s.to_str()).map(|stem| {
                        let location = file.display().to_string();
                        (
                            Found {
                                value: stem.to_owned(),
                                location: location.clone(),
                            },
                            Found {
                                value: format!("pyochain.{folder}.{stem}"),
                                location,
                            },
                        )
                    })
                })
                .unzip();

            let has_names = !file_names.is_empty();
            let names = std::iter::once(folder)
                .filter(|_| has_names)
                .map(|folder| Found {
                    value: folder.to_owned(),
                    location: path.display().to_string(),
                })
                .chain(file_names)
                .collect();
            (names, paths)
        })
        .unwrap_or_else(|| (Vec::new(), Vec::new()))
}

fn on_file(path: &Path) -> (Vec<Found>, Vec<Found>) {
    path.file_stem()
        .and_then(|s| s.to_str())
        .filter(|stem| *stem != "rs")
        .map_or((Vec::new(), Vec::new()), |stem| {
            (
                vec![Found {
                    value: stem.to_owned(),
                    location: path.display().to_string(),
                }],
                Vec::new(),
            )
        })
}

fn assert_subset(subset: &[Found], superset: &[Found], msg: &str) {
    let bad: Vec<&Found> = subset
        .iter()
        .filter(|f| !superset.iter().any(|g| g.value == f.value))
        .collect();
    if !bad.is_empty() {
        panic!("{msg}: {bad:?}");
    }
}
