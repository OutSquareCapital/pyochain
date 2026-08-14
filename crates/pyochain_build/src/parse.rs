use std::collections::HashSet;
use std::fs;
use std::ops::ControlFlow;
use std::path::{Components, Display, Path, PathBuf};
use syn::{Expr, Item, Lit, Meta, spanned::Spanned, visit::Visit};
use tap::Pipe;
use walkdir::WalkDir;
#[derive(Clone, Debug, Eq)]
pub(super) struct NormalizedPath {
    source: PathBuf,
    parent: Vec<String>,
    stem: String,
}
impl PartialEq for NormalizedPath {
    fn eq(&self, other: &Self) -> bool {
        self.parent == other.parent && self.stem == other.stem
    }
}
impl NormalizedPath {
    pub(super) fn display(&self) -> Display<'_> {
        self.source.display()
    }
}
#[derive(Default)]
struct RegisteredClassVisitor {
    classes: HashSet<String>,
}
impl RegisteredClassVisitor {
    fn visit(source: String) -> HashSet<String> {
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

pub(super) struct Root(PathBuf);
impl Root {
    pub(super) fn new(path: PathBuf) -> Self {
        Self(path)
    }
    pub(super) fn display(&self) -> Display<'_> {
        self.0.display()
    }
    pub(super) fn join(&self, path: &str) -> PathBuf {
        self.0.join(path)
    }

    pub(super) fn get_pyclasses(&self, lib_path: &str) -> Vec<PyClass> {
        self.join(lib_path)
            .pipe(fs::read_to_string)
            .unwrap()
            .pipe(RegisteredClassVisitor::visit)
            .pipe(|registered_classes| {
                self.iter_on_extension("rs")
                    .flat_map(|path: PathBuf| {
                        get_classes_from_file(&path, &registered_classes, self)
                    })
                    .collect::<Vec<_>>()
            })
    }

    pub(super) fn iter_on_extension(&self, extension: &str) -> impl Iterator<Item = PathBuf> {
        WalkDir::new(&self.0)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter(|entry| {
                entry.path().extension().and_then(|value| value.to_str()) == Some(extension)
            })
            .map(|entry| entry.into_path())
    }

    pub(super) fn make_relative<'a>(&self, path: &'a Path) -> RelativePath<'a> {
        RelativePath(path.strip_prefix(&self.0).unwrap())
    }
}

/// Only relative path can be normalized
pub(super) struct RelativePath<'a>(&'a Path);
impl<'a> RelativePath<'a> {
    pub(super) fn normalize(self) -> NormalizedPath {
        fn normalize_os_str(os_str: &std::ffi::OsStr) -> String {
            os_str.to_str().unwrap().trim_start_matches('_').to_string()
        }
        NormalizedPath {
            source: self.0.to_path_buf(),
            parent: self
                .components()
                .map(|component| component.as_os_str().pipe(normalize_os_str))
                .collect(),
            stem: self.0.file_stem().unwrap().pipe(normalize_os_str),
        }
    }
    pub(super) fn components(&self) -> Components<'_> {
        self.0.parent().unwrap().components()
    }
}
#[derive(Clone, Debug)]
pub(super) struct PyClass {
    pub(super) path: NormalizedPath,
    pub(super) line: usize,
    pub(super) rust_name: String,
    pub(super) python_name: String,
    pub(super) module: Option<String>,
}

impl PyClass {
    fn from_item(item: Item, relative: RelativePath<'_>) -> Option<Self> {
        let (attrs, ident, span) = match item {
            Item::Struct(item) => {
                let span = item.span();
                (item.attrs, item.ident, span)
            }
            Item::Enum(item) => {
                let span = item.span();
                (item.attrs, item.ident, span)
            }
            _ => return None,
        };
        let (name, module) = get_name_and_module_from_attrs(attrs)?;
        Some(Self {
            path: relative.normalize(),
            line: span.start().line,
            rust_name: ident.to_string(),
            python_name: name.unwrap_or_else(|| ident.to_string()),
            module,
        })
    }
}
#[inline]
fn get_classes_from_file(
    path: &Path,
    registered_classes: &HashSet<String>,
    root: &Root,
) -> Vec<PyClass> {
    let source = fs::read_to_string(&path).unwrap();
    syn::parse_file(&source)
        .unwrap()
        .items
        .into_iter()
        .filter_map(|item| PyClass::from_item(item, root.make_relative(&path)))
        .filter(|pyclass| registered_classes.contains(&pyclass.rust_name))
        .collect::<Vec<_>>()
}
fn get_name_and_module_from_attrs(
    attrs: Vec<syn::Attribute>,
) -> Option<(Option<String>, Option<String>)> {
    let meta = attrs
        .into_iter()
        .find(|attribute| attribute.path().is_ident("pyclass"))?
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        .expect("Failed to parse pyclass attributes");
    let values = meta
        .iter()
        .filter_map(|meta| match meta {
            Meta::NameValue(meta) => match (&meta.path, &meta.value) {
                (path, Expr::Lit(expression)) => match &expression.lit {
                    Lit::Str(value) if path.is_ident("name") => Some((Some(value.value()), None)),
                    Lit::Str(value) if path.is_ident("module") => Some((None, Some(value.value()))),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        })
        .try_fold((None, None), |(name, module), (new_name, new_module)| {
            let values = (name.or(new_name), module.or(new_module));
            match values {
                (Some(_), Some(_)) => ControlFlow::Break(values),
                _ => ControlFlow::Continue(values),
            }
        });
    match values {
        ControlFlow::Continue(values) | ControlFlow::Break(values) => Some(values),
    }
}
