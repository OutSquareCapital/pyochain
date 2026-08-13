use crate::parse::{self, PyClass};
use owo_colors::OwoColorize;
use std::{
    fmt::{self, Display},
    fs,
    path::{Path, PathBuf},
    process,
};

pub(super) fn run(stub_root: &Path, pyclasses: &[PyClass]) {
    let failures = get_failures(&pyclasses, &stub_root);
    let failures_len = failures.len();
    if failures_len == 0 {
        println!(
            "pyclass stub check passed ({} declarations)",
            pyclasses.len()
        )
    } else {
        let add_sep = || {
            anstream::eprintln!("\n============================================================");
        };
        add_sep();
        failures
            .into_iter()
            .for_each(|(index, failure)| failure.show(index));
        add_sep();
        anstream::eprintln!(
            "{}: {} issue(s) found in {} pyclass declaration(s).",
            "Pyclass / stub consistency check failed".red().bold(),
            failures_len.red().bold(),
            pyclasses.len().cyan(),
        );

        add_sep();
        process::exit(1);
    }
}
fn get_failures(pyclasses: &[PyClass], stub_root: &Path) -> Vec<(usize, Failure)> {
    pyclasses
        .iter()
        .filter_map(|pyclass| StubResult::new(pyclass, stub_root).into_failure())
        .enumerate()
        .collect::<Vec<_>>()
}
fn get_matching_stubs(pyclass: &PyClass, stub_root: &Path) -> Vec<Stub> {
    parse::files_with_extension(stub_root, "pyi")
        .into_iter()
        .filter(|path| path.file_stem().and_then(|stem| stem.to_str()) != Some("__init__"))
        .filter_map(|path| {
            fs::read_to_string(&path)
                .unwrap()
                .lines()
                .enumerate()
                .filter_map(|(line_index, line)| {
                    let name = line
                        .trim_start()
                        .strip_prefix("class ")
                        .or_else(|| line.trim_start().strip_prefix("type "))?
                        .chars()
                        .take_while(|character| {
                            character.is_ascii_alphanumeric() || *character == '_'
                        })
                        .collect::<String>();
                    (!name.is_empty()).then_some((name, line_index + 1))
                })
                .find(|(name, _)| name == &pyclass.python_name)
                .map(|(_, line)| Stub {
                    path: path.to_path_buf(),
                    normalized_path: parse::normalized_path(&path, stub_root),
                    module: stub_module(&path, stub_root),
                    line,
                })
        })
        .collect::<Vec<_>>()
}

fn stub_module(path: &Path, stub_root: &Path) -> String {
    let module_parts = path
        .strip_prefix(stub_root)
        .unwrap()
        .parent()
        .unwrap()
        .components()
        .map(|component| component.as_os_str().to_str().unwrap())
        .collect::<Vec<_>>()
        .join(".");
    format!("pyochain.{module_parts}")
}

enum StubResult<'a> {
    MissingModule(&'a PyClass),
    NoMatchingStub(&'a PyClass, &'a Path),
    DuplicateStub(&'a PyClass, Vec<Stub>),
    PathNotEq(&'a PyClass, Stub),
    ModuleNotEq(&'a PyClass, &'a String, Stub),
    Ok,
}
impl<'a> StubResult<'a> {
    fn new(pyclass: &'a PyClass, stub_root: &'a Path) -> Self {
        if pyclass.module.is_none() {
            Self::MissingModule(pyclass)
        } else {
            let mut matching_stubs = get_matching_stubs(pyclass, stub_root);
            if matching_stubs.is_empty() {
                Self::NoMatchingStub(pyclass, stub_root)
            } else if matching_stubs.len() > 1 {
                Self::DuplicateStub(pyclass, matching_stubs)
            } else {
                let stub = matching_stubs.remove(0);
                if pyclass.normalized_path != stub.normalized_path {
                    Self::PathNotEq(pyclass, stub)
                } else {
                    let module = pyclass.module.as_ref().unwrap();
                    (module != &stub.module)
                        .then(|| Self::ModuleNotEq(pyclass, module, stub))
                        .unwrap_or(Self::Ok)
                }
            }
        }
    }
    fn into_failure(self) -> Option<Failure> {
        match self {
            Self::Ok => None,
            Self::MissingModule(pyclass) => Some(Failure::missing_module(pyclass)),
            Self::NoMatchingStub(pyclass, stub_root) => {
                Some(Failure::missing_stub_class_declaration(pyclass, stub_root))
            }
            Self::DuplicateStub(pyclass, matching_stubs) => {
                let paths = matching_stubs
                    .iter()
                    .map(|stub| format!("{}:{}", stub.path.display(), stub.line))
                    .collect::<Vec<_>>()
                    .join(", ");
                Some(Failure::duplicate_stub_class_declaration(pyclass, &paths))
            }
            Self::PathNotEq(pyclass, stub) => Some(Failure::source_stub_mismatch(pyclass, &stub)),
            Self::ModuleNotEq(pyclass, module, stub) => {
                Some(Failure::module_mismatch(pyclass, module, &stub))
            }
        }
    }
}

enum FailureKind {
    MissingModuleDeclaration,
    MissingStubClassDeclaration,
    DuplicateStubClassDeclaration,
    SourceStubMismatch,
    ModuleMismatch,
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

struct Failure {
    kind: FailureKind,
    pyclass: PyClass,
    message: String,
}

struct Stub {
    path: PathBuf,
    normalized_path: parse::NormalizedPath,
    module: String,
    line: usize,
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

    fn source_stub_mismatch(pyclass: &PyClass, stub: &Stub) -> Self {
        Self::new(
            pyclass,
            FailureKind::SourceStubMismatch,
            format!(
                "The stub declaration is at {}:{}.
                    The Rust source file is: {}
                    The stub path must correspond to the Rust path, with optional prefixes.",
                stub.path.display(),
                stub.line,
                pyclass.source.display(),
            ),
        )
    }

    fn module_mismatch(pyclass: &PyClass, module: &str, stub: &Stub) -> Self {
        Self::new(
            pyclass,
            FailureKind::ModuleMismatch,
            format!(
                "Rust declares: module = \"{module}\"\n       Stub declaration: {}:{}\n       Module implied by stub path: \"{}\"\n",
                stub.path.display(),
                stub.line,
                stub.module
            ),
        )
    }

    fn show(&self, index: usize) {
        anstream::eprintln!(
            "\n{:>3}. {}",
            index + 1,
            self.kind.to_string().yellow().bold(),
        );
        anstream::eprintln!("     Rust declaration: `{}`", self.pyclass.rust_name.cyan());
        anstream::eprintln!(
            "     Python class:      `{}`",
            self.pyclass.python_name.cyan()
        );
        anstream::eprintln!(
            "     Rust location:     {}:{}",
            self.pyclass.source.display().to_string().yellow(),
            self.pyclass.line.to_string().yellow(),
        );
        self.message
            .lines()
            .for_each(|line| anstream::eprintln!("     {line}"));
    }
}
