use crate::parse::PyClass;
use crate::paths;
use owo_colors::OwoColorize;
use std::{
    fmt::{self, Display},
    fs,
    iter::Peekable,
    path::Path,
    process,
};
use tap::Pipe;
/// Full path and name of a Python class declaration in a stub file.
pub type Reference = (String, String);
pub(super) fn run(stub_root: &paths::Root, pyclasses: &[PyClass]) -> Vec<Reference> {
    let (references, failures) = pyclasses.iter().fold(
        (Vec::new(), Vec::new()),
        |(mut references, mut failures), pyclass| {
            match StubResult::new(pyclass, stub_root) {
                StubResult::Ok { name, full_path } => references.push((name, full_path)),
                failure => failures.push((pyclass, failure)),
            }
            (references, failures)
        },
    );
    if failures.is_empty() {
        let msg = "pyclass stub check passed!";
        println!("{} ({} declarations)", msg.green().bold(), pyclasses.len());
        references
    } else {
        failures
            .into_iter()
            .enumerate()
            .peekable()
            .pipe(|failures| show_failures(failures, pyclasses));
        process::exit(1);
    }
}
fn show_failures<'a>(
    failures: Peekable<impl Iterator<Item = (usize, (&'a PyClass, StubResult))> + 'a>,
    pyclasses: &[PyClass],
) {
    let add_sep = || {
        anstream::eprintln!("\n============================================================");
    };
    add_sep();
    let count = failures
        .inspect(|(index, (pyclass, failure))| failure.show(*index, pyclass))
        .count();
    add_sep();
    anstream::eprintln!(
        "{}: {} issue(s) found in {} pyclass declaration(s).",
        "Pyclass / stub consistency check failed".red().bold(),
        count.red().bold(),
        pyclasses.len().cyan(),
    );
    add_sep();
}

impl Display for PyClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "     {:<20}`{}`",
            "Rust declaration:",
            self.rust_name.cyan()
        )?;
        writeln!(
            formatter,
            "     {:<20}`{}`",
            "Python class:",
            self.python_name.cyan()
        )?;
        write!(
            formatter,
            "     {:<20}{}:{}",
            "Rust location:",
            self.path.display().to_string().yellow(),
            self.line.to_string().yellow()
        )
    }
}

enum StubResult {
    Ok { name: String, full_path: String },
    Missing(String),
    Duplicated(Vec<Stub>),
    NoCorrespondance(Stub),
}

impl StubResult {
    fn new(pyclass: &PyClass, stub_root: &paths::Root) -> Self {
        let mut matching_stubs = get_matching_stubs(pyclass, stub_root);
        match matching_stubs.len() {
            0 => Self::Missing(stub_root.display().to_string()),
            1 => {
                let stub = matching_stubs.remove(0);
                if pyclass.path == stub.path {
                    Self::Ok {
                        name: pyclass.python_name.clone(),
                        full_path: format!("{}.{}", stub.module, pyclass.python_name),
                    }
                } else {
                    Self::NoCorrespondance(stub)
                }
            }
            _ => Self::Duplicated(matching_stubs),
        }
    }
    fn show(&self, index: usize, pyclass: &PyClass) {
        anstream::eprintln!("\n{:>3}. {}", index + 1, self.title().yellow().bold(),);
        anstream::eprint!("{}\n", pyclass);
        match self {
            Self::Ok { .. } => {}
            Self::Missing(root) => show_detail(format_args!(
                "No `.pyi` file under {root} declares Python class `{}`.",
                pyclass.python_name
            )),
            Self::Duplicated(stubs) => {
                show_detail(format_args!(
                    "Python class `{}` is declared in more than one stub:",
                    pyclass.python_name
                ));
                stubs
                    .iter()
                    .for_each(|stub| show_field("Stub declaration:", stub));
            }
            Self::NoCorrespondance(stub) => {
                show_field("Stub location:", stub);
                show_detail(
                    "The stub path must correspond to the Rust path, with optional prefixes.",
                );
            }
        }
    }
    fn title(&self) -> &'static str {
        match self {
            Self::Ok { .. } => "Stub check passed",
            Self::Missing(_) => "missing stub class declaration",
            Self::Duplicated(_) => "duplicate stub class declaration",
            Self::NoCorrespondance(_) => "Rust source file does not match stub file",
        }
    }
}
struct Stub {
    path: paths::Normalized,
    module: String,
    line: usize,
}

impl Display for Stub {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.path.display(), self.line)
    }
}

fn show_field(label: &str, value: impl Display) {
    anstream::eprintln!("     {label:<20}{value}");
}

fn show_detail(detail: impl Display) {
    anstream::eprintln!("     {detail}");
}

fn get_matching_stubs(pyclass: &PyClass, stub_root: &paths::Root) -> Vec<Stub> {
    stub_root
        .iter_on_extension("pyi")
        .filter(|path| path.file_stem().and_then(|stem| stem.to_str()) != Some("__init__"))
        .filter_map(|path| {
            path.pipe_ref(fs::read_to_string)
                .unwrap()
                .lines()
                .enumerate()
                .find_map(|(index, line)| name_from_line(index, line, &pyclass.python_name))
                .map(|line| Stub {
                    path: stub_root.make_relative(&path).normalize(),
                    module: stub_module(&path, stub_root),
                    line,
                })
        })
        .collect::<Vec<_>>()
}

fn stub_module(path: &Path, root: &paths::Root) -> String {
    let module_parts = root
        .make_relative(path)
        .0
        .with_extension("")
        .components()
        .map(|component| component.as_os_str().to_str().unwrap())
        .collect::<Vec<_>>()
        .join(".");
    format!("pyochain.{module_parts}")
}

fn name_from_line(index: usize, line: &str, python_name: &str) -> Option<usize> {
    line.trim_start()
        .strip_prefix("class ")
        .or_else(|| line.trim_start().strip_prefix("type "))?
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect::<String>()
        .pipe(|name| name == python_name)
        .then(|| index + 1)
}
