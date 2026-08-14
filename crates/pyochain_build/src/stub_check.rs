use crate::parse::{NormalizedPath, PyClass, Root};
use owo_colors::OwoColorize;
use std::{
    fmt::{self, Display},
    fs,
    iter::Peekable,
    path::Path,
    process,
};
use tap::Pipe;

pub(super) fn run(stub_root: &Root, pyclasses: &[PyClass]) {
    let mut failures = get_failures(&pyclasses, &stub_root);
    if failures.peek().is_none() {
        let msg = "pyclass stub check passed!";
        println!("{} ({} declarations)", msg.green().bold(), pyclasses.len());
    } else {
        show_failures(failures, &pyclasses);
        process::exit(1);
    }
}
fn show_failures<'a>(
    failures: Peekable<impl Iterator<Item = (usize, (&'a PyClass, CheckErr))> + 'a>,
    pyclasses: &[PyClass],
) {
    let add_sep = || {
        anstream::eprintln!("\n============================================================");
    };
    add_sep();
    let count = failures.count();
    add_sep();
    anstream::eprintln!(
        "{}: {} issue(s) found in {} pyclass declaration(s).",
        "Pyclass / stub consistency check failed".red().bold(),
        count.red().bold(),
        pyclasses.len().cyan(),
    );
    add_sep();
}
fn get_failures<'a>(
    pyclasses: &'a [PyClass],
    stub_root: &'a Root,
) -> Peekable<impl Iterator<Item = (usize, (&'a PyClass, CheckErr))> + 'a> {
    pyclasses
        .iter()
        .filter_map(|pyclass| CheckErr::new(pyclass, stub_root).map(|failure| (pyclass, failure)))
        .enumerate()
        .inspect(|(index, (pyclass, failure))| failure.show(*index, pyclass))
        .peekable()
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

enum CheckErr {
    MissingModule,
    MissingStub { root: String },
    DuplicateStub { stubs: Vec<Stub> },
    SourceStub { stub: Stub },
    ModuleNotEq { module: String, stub: Stub },
}

impl CheckErr {
    fn new(pyclass: &PyClass, stub_root: &Root) -> Option<Self> {
        match pyclass.module.as_ref() {
            None => Some(Self::MissingModule),
            Some(module) => {
                let mut matching_stubs = get_matching_stubs(pyclass, stub_root);
                match matching_stubs.len() {
                    0 => Some(Self::MissingStub {
                        root: stub_root.display().to_string(),
                    }),
                    1 => {
                        let stub = matching_stubs.remove(0);
                        if pyclass.path != stub.path {
                            Some(Self::SourceStub { stub: stub })
                        } else {
                            if module != &stub.module {
                                Some(Self::ModuleNotEq {
                                    module: module.to_string(),
                                    stub: stub,
                                })
                            } else {
                                None
                            }
                        }
                    }
                    _ => Some(Self::DuplicateStub {
                        stubs: matching_stubs,
                    }),
                }
            }
        }
    }

    fn show(&self, index: usize, pyclass: &PyClass) {
        anstream::eprintln!("\n{:>3}. {}", index + 1, self.title().yellow().bold(),);
        anstream::eprint!("{}\n", pyclass);
        match self {
            Self::MissingModule => {
                show_detail("The `#[pyclass(...)]` attribute has no `module = \"...\"` entry.")
            }
            Self::MissingStub { root } => show_detail(format_args!(
                "No `.pyi` file under {root} declares Python class `{}`.",
                pyclass.python_name
            )),
            Self::DuplicateStub { stubs } => {
                show_detail(format_args!(
                    "Python class `{}` is declared in more than one stub:",
                    pyclass.python_name
                ));
                stubs
                    .iter()
                    .for_each(|stub| show_field("Stub declaration:", stub));
            }
            Self::SourceStub { stub } => {
                show_field("Stub location:", stub);
                show_detail(
                    "The stub path must correspond to the Rust path, with optional prefixes.",
                );
            }
            Self::ModuleNotEq { module, stub } => {
                show_field("Declared module:", format_args!("`{module}`"));
                show_field("Stub declaration:", stub);
                show_field("Implied module:", format_args!("`{}`", stub.module));
            }
        }
    }
    fn title(&self) -> &'static str {
        match self {
            Self::MissingModule => "missing module declaration",
            Self::MissingStub { .. } => "missing stub class declaration",
            Self::DuplicateStub { .. } => "duplicate stub class declaration",
            Self::SourceStub { .. } => "Rust source file does not match stub file",
            Self::ModuleNotEq { .. } => "Rust module does not match stub module",
        }
    }
}
struct Stub {
    path: NormalizedPath,
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

fn get_matching_stubs(pyclass: &PyClass, stub_root: &Root) -> Vec<Stub> {
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

fn stub_module(path: &Path, root: &Root) -> String {
    let module_parts = root
        .make_relative(path)
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
