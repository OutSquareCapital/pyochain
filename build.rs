use std::{
    fmt::{self, Display},
    fs,
    path::{Path, PathBuf},
    process,
};

use owo_colors::OwoColorize;
use syn::{Expr, Item, Lit, Meta, spanned::Spanned};
use tap::Pipe;
struct Failure {
    kind: FailureKind,
    pyclass: PyClass,
    message: String,
}

#[derive(Clone, Debug)]
struct PyClass {
    source: PathBuf,
    normalized_path: NormalizedPath,
    line: usize,
    rust_name: String,
    python_name: String,
    module: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormalizedPath {
    parent: Vec<String>,
    stem: String,
}

struct Stub {
    path: PathBuf,
    normalized_path: NormalizedPath,
    module: String,
    classes: Vec<(String, usize)>,
}

struct StubMatch<'a> {
    stub: &'a Stub,
    line: usize,
}

enum FailureKind {
    MissingModuleDeclaration,
    MissingStubClassDeclaration,
    DuplicateStubClassDeclaration,
    SourceStubMismatch,
    ModuleMismatch,
}
fn main() {
    if std::env::var_os("CARGO_FEATURE_STUB_CHECK").is_some() {
        run_stub_check();
    }
}
fn run_stub_check() {
    anstream::ColorChoice::Always.write_global();

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=pyochain");

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_root = root.join("src");
    let stub_root = root.join("pyochain");
    let pyclasses = get_pyclasses(&source_root);
    let stubs = get_stubs(&stub_root);
    let failures = get_failures(&pyclasses, &stubs, &stub_root);
    show_output(&failures, &pyclasses);
}
fn show_output(failures: &[Failure], pyclasses: &[PyClass]) {
    if failures.is_empty() {
        println!(
            "cargo:warning=pyclass stub check passed ({} declarations)",
            pyclasses.len()
        )
    } else {
        anstream::eprintln!("\n============================================================");
        anstream::eprintln!(
            "{}: {} issue(s) found in {} pyclass declaration(s).",
            "Pyclass / stub consistency check failed".red().bold(),
            failures.len().red().bold(),
            pyclasses.len().cyan(),
        );
        anstream::eprintln!("============================================================");
        failures.iter().enumerate().for_each(|(index, failure)| {
            anstream::eprintln!(
                "\n{:>3}. {}",
                index + 1,
                failure.kind.to_string().yellow().bold(),
            );
            anstream::eprintln!(
                "     Rust declaration: `{}`",
                failure.pyclass.rust_name.cyan()
            );
            anstream::eprintln!(
                "     Python class:      `{}`",
                failure.pyclass.python_name.cyan()
            );
            anstream::eprintln!(
                "     Rust location:     {}:{}",
                failure.pyclass.source.display().to_string().yellow(),
                failure.pyclass.line.to_string().yellow(),
            );
            failure
                .message
                .lines()
                .for_each(|line| anstream::eprintln!("     {line}"));
        });
        anstream::eprintln!("\n============================================================");
        process::exit(1);
    }
}
fn get_failures(pyclasses: &[PyClass], stubs: &[Stub], stub_root: &Path) -> Vec<Failure> {
    pyclasses
        .iter()
        .flat_map(|pyclass| {
            let matching_stubs = stubs
                .iter()
                .filter_map(|stub| {
                    stub.classes
                        .iter()
                        .find(|(name, _)| name == &pyclass.python_name)
                        .map(|(_, line)| StubMatch { stub, line: *line })
                })
                .collect::<Vec<_>>();
            let mut failures = Vec::new();

            if pyclass.module.is_none() {
                failures.push(Failure::missing_module(pyclass))
            }

            if matching_stubs.is_empty() {
                failures.push(Failure::missing_stub_class_declaration(pyclass, stub_root));
                failures
            } else {
                if matching_stubs.len() > 1 {
                    let paths = matching_stubs
                        .iter()
                        .map(|stub_match| {
                            format!("{}:{}", stub_match.stub.path.display(), stub_match.line)
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    failures.push(Failure::duplicate_stub_class_declaration(pyclass, &paths));
                }

                matching_stubs.iter().for_each(|stub_match| {
                    if !(pyclass.normalized_path == stub_match.stub.normalized_path) {
                        failures.push(Failure::source_stub_mismatch(pyclass, stub_match));
                    }

                    if let Some(module) = &pyclass.module {
                        if module != &stub_match.stub.module {
                            failures.push(Failure::module_mismatch(
                                pyclass,
                                module,
                                stub_match,
                                &stub_match.stub.module,
                            ));
                        }
                    }
                });

                failures
            }
        })
        .collect::<Vec<_>>()
}

fn normalized_path(path: &Path, root: &Path) -> NormalizedPath {
    let relative = path.strip_prefix(root).unwrap();
    NormalizedPath {
        parent: relative
            .parent()
            .unwrap()
            .components()
            .map(|component| component.as_os_str().pipe(normalize_os_str))
            .collect(),
        stem: relative.file_stem().unwrap().pipe(normalize_os_str),
    }
}
fn normalize_os_str(os_str: &std::ffi::OsStr) -> String {
    os_str.to_str().unwrap().trim_start_matches('_').to_string()
}
fn get_pyclasses(source_root: &Path) -> Vec<PyClass> {
    files_with_extension(&source_root, "rs")
        .into_iter()
        .flat_map(|path| {
            let source = fs::read_to_string(&path).unwrap();
            syn::parse_file(&source)
                .unwrap()
                .items
                .into_iter()
                .filter_map(|item| PyClass::from_item(item, &path, source_root))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}
fn get_stubs(stub_root: &Path) -> Vec<Stub> {
    files_with_extension(&stub_root, "pyi")
        .into_iter()
        .filter(|path| path.file_stem().and_then(|stem| stem.to_str()) != Some("__init__"))
        .map(|path| {
            let source = fs::read_to_string(&path).unwrap();
            let classes = source
                .lines()
                .enumerate()
                .filter_map(|(line_index, line)| {
                    let declaration = line
                        .trim_start()
                        .strip_prefix("class ")
                        .or_else(|| line.trim_start().strip_prefix("type "))?;
                    let name = declaration
                        .chars()
                        .take_while(|character| {
                            character.is_ascii_alphanumeric() || *character == '_'
                        })
                        .collect::<String>();
                    (!name.is_empty()).then_some((name, line_index + 1))
                })
                .collect();
            Stub {
                path: path.to_path_buf(),
                normalized_path: normalized_path(&path, stub_root),
                module: stub_module(&path, stub_root),
                classes,
            }
        })
        .collect::<Vec<_>>()
}

fn stub_module(path: &Path, stub_root: &Path) -> String {
    let components = path
        .strip_prefix(stub_root)
        .unwrap()
        .components()
        .map(|component| component.as_os_str().to_str().unwrap())
        .collect::<Vec<_>>();
    let module_parts = components[..components.len() - 1]
        .iter()
        .chain(std::iter::once(
            &components[components.len() - 1].trim_end_matches(".pyi"),
        ))
        .copied()
        .collect::<Vec<_>>()
        .join(".");
    format!("pyochain.{module_parts}")
}

fn files_with_extension(root: &Path, extension: &str) -> Vec<PathBuf> {
    fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .flat_map(|path| {
            if path.is_dir() {
                files_with_extension(&path, extension)
            } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
                vec![path]
            } else {
                Vec::new()
            }
        })
        .collect()
}

impl PyClass {
    fn from_item(item: Item, path: &Path, root: &Path) -> Option<Self> {
        let (attrs, ident, span) = match item {
            Item::Struct(item) => (item.attrs.clone(), item.ident.clone(), item.span()),
            Item::Enum(item) => (item.attrs.clone(), item.ident.clone(), item.span()),
            _ => return None,
        };
        let pyclass = attrs
            .into_iter()
            .find(|attribute| attribute.path().is_ident("pyclass"))?;
        let meta = pyclass
            .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
            .unwrap();
        let name = meta.iter().find_map(|meta| match meta {
            Meta::NameValue(value) if value.path.is_ident("name") => match &value.value {
                Expr::Lit(expression) => match &expression.lit {
                    Lit::Str(value) => Some(value.value()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        });
        let module = meta.iter().find_map(|meta| match meta {
            Meta::NameValue(value) if value.path.is_ident("module") => match &value.value {
                Expr::Lit(expression) => match &expression.lit {
                    Lit::Str(value) => Some(value.value()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        });
        Some(Self {
            source: path.to_path_buf(),
            normalized_path: normalized_path(path, root),
            line: span.start().line,
            rust_name: ident.to_string(),
            python_name: name.unwrap_or_else(|| ident.to_string()),
            module,
        })
    }
}

impl Display for FailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::MissingModuleDeclaration => "missing module declaration",
            Self::MissingStubClassDeclaration => "missing stub class declaration",
            Self::DuplicateStubClassDeclaration => "duplicate stub class declaration",
            Self::SourceStubMismatch => "Rust source file does not match stub file",
            Self::ModuleMismatch => "Rust module does not match stub module",
        })
    }
}

impl Failure {
    fn new(pyclass: &PyClass, kind: FailureKind, message: String) -> Self {
        Self {
            kind,
            pyclass: pyclass.clone(),
            message,
        }
    }
    fn missing_module(pyclass: &PyClass) -> Self {
        Self::new(
            pyclass,
            FailureKind::MissingModuleDeclaration,
            "The `#[pyclass(...)]` attribute has no `module = \"...\"` entry.".to_string(),
        )
    }
    fn missing_stub_class_declaration(pyclass: &PyClass, stub_root: &Path) -> Self {
        Self::new(
            pyclass,
            FailureKind::MissingStubClassDeclaration,
            format!(
                "No `.pyi` file under {} declares Python class `{}`.",
                stub_root.display(),
                pyclass.python_name,
            ),
        )
    }
    fn duplicate_stub_class_declaration(pyclass: &PyClass, paths: &str) -> Self {
        Self::new(
            pyclass,
            FailureKind::DuplicateStubClassDeclaration,
            format!(
                "Python class `{}` is declared in more than one stub:\n       {}",
                pyclass.python_name, paths,
            ),
        )
    }
    fn source_stub_mismatch(pyclass: &PyClass, stub_match: &StubMatch) -> Self {
        Self::new(
            pyclass,
            FailureKind::SourceStubMismatch,
            format!(
                "The stub declaration is at {}:{}.
                    The Rust source file is: {}
                    The stub path must correspond to the Rust path, with optional prefixes.",
                stub_match.stub.path.display(),
                stub_match.line,
                pyclass.source.display(),
            ),
        )
    }
    fn module_mismatch(
        pyclass: &PyClass,
        module: &str,
        stub_match: &StubMatch,
        stub_module: &str,
    ) -> Self {
        Self::new(
            pyclass,
            FailureKind::ModuleMismatch,
            format!(
                "Rust declares: module = \"{module}\"\n       Stub declaration: {}:{}\n       Module implied by stub path: \"{stub_module}\"\n       Expected Rust declaration: module = \"{stub_module}\"",
                stub_match.stub.path.display(),
                stub_match.line,
            ),
        )
    }
}
