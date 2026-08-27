use comfy_table::{Attribute, Cell, Color, ContentArrangement, Table};
use owo_colors::OwoColorize;
use ruff_python_ast::{Expr, Stmt, statement_visitor, statement_visitor::StatementVisitor};
use ruff_python_parser::parse_module;
use ruff_text_size::TextSize;
use std::{
    error::Error,
    ffi::OsStr,
    fmt, fs, io,
    path::{Path, PathBuf},
};

use crate::paths;

const SKIP_DECORATORS: [&str; 4] = ["overload", "override", "wraps", "property"];
const BLOCK_MARKER: &str = "```";
const RICH_TABLE_STYLE: &str = "││━━┡━╇┩│    ┳┻┏┓┗┛";

#[derive(Clone, Copy)]
enum DiagnosticKind {
    Missing,
    NoDoctest,
    Unclosed,
    ClosedNotOpened,
}

impl fmt::Display for DiagnosticKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Missing => "Missing docstring",
            Self::NoDoctest => "No ```python block found in docstring",
            Self::Unclosed => "Unclosed ``` block",
            Self::ClosedNotOpened => "Closing ``` block without opening",
        })
    }
}

struct Diagnostic {
    line_no: usize,
    kind: DiagnosticKind,
}

struct DocstringError {
    file_path: PathBuf,
    function_name: String,
    error_line_no: usize,
    diagnostics: Vec<Diagnostic>,
}

struct FileChecker<'source> {
    file_path: &'source Path,
    source: &'source str,
    errors: Vec<DocstringError>,
}

impl FileChecker<'_> {
    fn check_node(
        &mut self,
        name: &str,
        node_start: TextSize,
        decorators: &[ruff_python_ast::Decorator],
        body: &[Stmt],
        is_function: bool,
    ) {
        if has_skip_decorator(decorators) {
            return;
        }

        let node_line = line_number(self.source, node_start);
        match get_docstring(body) {
            None if is_function => self.errors.push(DocstringError {
                file_path: self.file_path.to_path_buf(),
                function_name: name.to_string(),
                error_line_no: node_line,
                diagnostics: vec![Diagnostic {
                    line_no: node_line,
                    kind: DiagnosticKind::Missing,
                }],
            }),
            None => {}
            Some((content, content_start)) => {
                let start_line = line_number(self.source, content_start);
                let mut diagnostics = check_code_blocks(content, start_line);
                if !has_python_block(content) {
                    diagnostics.insert(
                        0,
                        Diagnostic {
                            line_no: start_line,
                            kind: DiagnosticKind::NoDoctest,
                        },
                    );
                }
                if !diagnostics.is_empty() {
                    let error_line_no = diagnostics.first().expect("non-empty diagnostics").line_no;
                    self.errors.push(DocstringError {
                        file_path: self.file_path.to_path_buf(),
                        function_name: name.to_string(),
                        error_line_no,
                        diagnostics,
                    });
                }
            }
        }
    }
}

impl<'ast> StatementVisitor<'ast> for FileChecker<'_> {
    fn visit_stmt(&mut self, statement: &'ast Stmt) {
        match statement {
            Stmt::FunctionDef(function) => self.check_node(
                function.name.as_str(),
                function.range.start(),
                &function.decorator_list,
                &function.body,
                true,
            ),
            Stmt::ClassDef(class) => self.check_node(
                class.name.as_str(),
                class.range.start(),
                &class.decorator_list,
                &class.body,
                false,
            ),
            _ => {}
        }
        statement_visitor::walk_stmt(self, statement);
    }
}

pub(super) fn run(stubs: &paths::Related) -> Result<(), Box<dyn Error>> {
    let files = stubs
        .iter()
        .filter(|path| path.file_name() != Some(OsStr::new("_types.pyi")))
        .collect::<Vec<_>>();
    anstream::eprintln!("Checking docstrings for properly closed code blocks...");
    anstream::eprintln!("Checking {} pyi files...", files.len());

    let errors = files.iter().try_fold(Vec::new(), |mut errors, path| {
        errors.extend(check_file(path)?);
        Ok::<_, Box<dyn Error>>(errors)
    })?;

    if errors.is_empty() {
        anstream::eprintln!("{}", "[OK] No issues found!".green());
    } else {
        show_errors(&errors, stubs);
        anstream::eprintln!(
            "{}",
            format!("[WARNING] Found {} issue(s)", errors.len())
                .yellow()
                .bold()
        );
    }
    Ok(())
}

fn check_file(file_path: &Path) -> Result<Vec<DocstringError>, Box<dyn Error>> {
    let source = fs::read_to_string(file_path)?;
    let parsed = parse_module(&source).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse {}: {error}", file_path.display()),
        )
    })?;
    let mut checker = FileChecker {
        file_path,
        source: &source,
        errors: Vec::new(),
    };
    checker.visit_body(parsed.suite());
    Ok(checker.errors)
}

fn show_errors(errors: &[DocstringError], stubs: &paths::Related) {
    let mut table = Table::new();
    table
        .load_preset(RICH_TABLE_STYLE)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("File").add_attribute(Attribute::Bold),
            Cell::new("Function").add_attribute(Attribute::Bold),
            Cell::new("Error").add_attribute(Attribute::Bold),
        ])
        .add_rows(errors.iter().map(|error| {
            let path = format!(
                "{}:{}",
                stubs.make_relative(&error.file_path).0.display(),
                error.error_line_no
            );
            let diagnostics = error
                .diagnostics
                .iter()
                .map(|diagnostic| diagnostic.kind.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            vec![
                Cell::new(path).fg(Color::Cyan),
                Cell::new(&error.function_name).fg(Color::Magenta),
                Cell::new(diagnostics).fg(Color::Red),
            ]
        }));
    anstream::eprintln!("{}", "Issues Found".bold());
    anstream::eprintln!("{table}");
}

fn get_docstring(body: &[Stmt]) -> Option<(&str, TextSize)> {
    let Some(Stmt::Expr(expression)) = body.first() else {
        return None;
    };
    let Expr::StringLiteral(string) = expression.value.as_ref() else {
        return None;
    };
    Some((string.value.to_str(), string.range.start()))
}

fn has_skip_decorator(decorators: &[ruff_python_ast::Decorator]) -> bool {
    decorators
        .iter()
        .any(|decorator| match &decorator.expression {
            Expr::Name(name) => SKIP_DECORATORS.contains(&name.id.as_str()),
            Expr::Attribute(attribute) => SKIP_DECORATORS.contains(&attribute.attr.as_str()),
            _ => false,
        })
}

fn check_code_blocks(docstring: &str, start_line: usize) -> Vec<Diagnostic> {
    let (open_blocks, mut diagnostics) = docstring.lines().enumerate().fold(
        (Vec::new(), Vec::new()),
        |(mut open_blocks, mut diagnostics), (line_num, line)| {
            let stripped = line.trim_start();
            if stripped.starts_with(BLOCK_MARKER) {
                if stripped == BLOCK_MARKER {
                    if open_blocks.pop().is_none() {
                        diagnostics.push(Diagnostic {
                            line_no: start_line + line_num,
                            kind: DiagnosticKind::ClosedNotOpened,
                        });
                    }
                } else {
                    open_blocks.push(line_num + 1);
                }
            }
            (open_blocks, diagnostics)
        },
    );
    diagnostics.extend(open_blocks.into_iter().map(|line_num| Diagnostic {
        line_no: start_line + line_num - 1,
        kind: DiagnosticKind::Unclosed,
    }));
    diagnostics
}

fn has_python_block(docstring: &str) -> bool {
    docstring.lines().any(|line| {
        let stripped = line.trim_start();
        stripped.starts_with(BLOCK_MARKER) && line.contains("python")
    })
}

fn line_number(source: &str, offset: TextSize) -> usize {
    source[..offset.to_usize()].split('\n').count()
}
