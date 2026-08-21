use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};
use syn::{Expr, Item, Lit, Meta, spanned::Spanned, visit::Visit};
use tap::Pipe;

use crate::paths;

#[derive(Clone, Debug)]
pub(super) struct PyClass {
    pub(super) path: paths::Normalized,
    pub(super) line: usize,
    pub(super) rust_name: String,
    pub(super) python_name: String,
}

impl PyClass {
    pub fn from_item(item: Item, relative: paths::Relative<'_>) -> Option<Self> {
        let (attrs, ident, span) = get_infos_from_item(item)?;
        let name = get_name_from_attrs(attrs).unwrap_or_else(|| ident.to_string());
        Some(Self {
            path: relative.normalize(),
            line: span.start().line,
            rust_name: ident.to_string(),
            python_name: name,
        })
    }

    pub fn document_path(&self) -> String {
        format!("reference/{}.md", self.python_name.to_lowercase())
    }

    pub fn check(
        &self,
        stub_root: &paths::Related,
        docs_dir: &Path,
    ) -> Result<(PathBuf, String), String> {
        let stubs = get_matching_stubs(&self.python_name, stub_root)
            .map_err(|error| format!("Failed to read stub files: {error}"))?;
        let detail = match stubs.as_slice() {
            [] => Err(format!(
                "missing stub class declaration\nNo `.pyi` file under {} declares Python class `{}`.",
                stub_root.display(),
                self.python_name
            )),
            [stub] if self.path != stub.path => Err(format!(
                "Rust source file does not match stub file\nStub location: {}:{}\nThe stub path must correspond to the Rust path, with optional prefixes.",
                stub.path.display(),
                stub.line
            )),
            [stub] => {
                let path = self.document_path();
                Ok((
                    docs_dir.join(path.trim_start_matches("reference/")),
                    format!(
                        "# {}\n\n::: {}.{}\n",
                        self.python_name, stub.module, self.python_name
                    ),
                ))
            }
            stubs => Err(format!(
                "duplicate stub class declaration\nPython class `{}` is declared in more than one stub:\n{}",
                self.python_name,
                stubs
                    .iter()
                    .map(|stub| format!("Stub declaration: {}:{}", stub.path.display(), stub.line))
                    .collect::<Vec<_>>()
                    .join("\n")
            )),
        };
        detail.map_err(|detail| {
            format!(
                "{detail}\nRust declaration: `{}`\nPython class: `{}`\nRust location: {}:{}",
                self.rust_name,
                self.python_name,
                self.path.display(),
                self.line
            )
        })
    }
}

fn get_infos_from_item(item: Item) -> Option<(Vec<syn::Attribute>, syn::Ident, proc_macro2::Span)> {
    match item {
        Item::Struct(item) => {
            let span = item.span();
            Some((item.attrs, item.ident, span))
        }
        Item::Enum(item) => {
            let span = item.span();
            Some((item.attrs, item.ident, span))
        }
        _ => None,
    }
}
fn get_name_from_attrs(attrs: Vec<syn::Attribute>) -> Option<String> {
    let meta = attrs
        .into_iter()
        .find(|attribute| attribute.path().is_ident("pyclass"))?
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        .expect("Failed to parse pyclass attributes");
    meta.iter().find_map(|meta| match meta {
        Meta::NameValue(meta) => match (&meta.path, &meta.value) {
            (path, Expr::Lit(expression)) if path.is_ident("name") => match &expression.lit {
                Lit::Str(value) => Some(value.value()),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    })
}
struct Stub {
    path: paths::Normalized,
    module: String,
    line: usize,
}

fn get_matching_stubs(
    python_name: &str,
    stub_root: &paths::Related,
) -> Result<Vec<Stub>, std::io::Error> {
    stub_root
        .iter()
        .filter(|path| path.file_stem().and_then(|stem| stem.to_str()) != Some("__init__"))
        .filter_map(|path| {
            path.pipe_ref(fs::read_to_string)
                .map(|source| {
                    source
                        .lines()
                        .enumerate()
                        .find_map(|(index, line)| name_from_line(index, line, python_name))
                        .map(|line| Stub {
                            path: stub_root.make_relative(&path).normalize(),
                            module: stub_module(&path, stub_root),
                            line,
                        })
                })
                .transpose()
        })
        .collect::<Result<Vec<Stub>, std::io::Error>>()
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
fn stub_module(path: &Path, root: &paths::Related) -> String {
    root.make_relative(path)
        .0
        .with_extension("")
        .components()
        .map(|component| {
            component
                .as_os_str()
                .to_str()
                .expect("Expected valid unicode path component")
        })
        .collect::<Vec<_>>()
        .join(".")
        .pipe(|module_parts| format!("pyochain.{module_parts}"))
}

#[derive(Default)]
pub(super) struct RegisteredClassVisitor {
    classes: HashSet<String>,
}
impl RegisteredClassVisitor {
    pub fn visit(source: String) -> HashSet<String> {
        let mut visitor = RegisteredClassVisitor::default();
        visitor.visit_file(&syn::parse_file(&source).unwrap());
        visitor.classes
    }
}
impl<'ast> Visit<'ast> for RegisteredClassVisitor {
    fn visit_expr_method_call(&mut self, expression: &'ast syn::ExprMethodCall) {
        if expression.method == "add_class" {
            if let Some(syn::GenericArgument::Type(syn::Type::Path(path))) = expression
                .turbofish
                .as_ref()
                .and_then(|turbofish| turbofish.args.first())
            {
                if let Some(segment) = path.path.segments.last() {
                    self.classes.insert(segment.ident.to_string());
                }
            }
        }
        syn::visit::visit_expr_method_call(self, expression);
    }
}
