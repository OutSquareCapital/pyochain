use std::fs;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use syn::{Expr, Item, Lit, Meta, spanned::Spanned, visit::Visit};
use tap::Pipe;
#[derive(Clone, Debug)]
pub(super) struct PyClass {
    pub(super) source: PathBuf,
    pub(super) normalized_path: NormalizedPath,
    pub(super) line: usize,
    pub(super) rust_name: String,
    pub(super) python_name: String,
    pub(super) module: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct NormalizedPath {
    parent: Vec<String>,
    stem: String,
}
#[derive(Default)]
struct RegisteredClassVisitor {
    classes: Vec<String>,
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
                    self.classes.push(segment.ident.to_string());
                }
            }
        }
        syn::visit::visit_expr_method_call(self, expression);
    }
}
pub(super) fn get_pyclasses(source_root: &Path, lib_path: &Path) -> Vec<PyClass> {
    let source = fs::read_to_string(lib_path).unwrap();
    let mut visitor = RegisteredClassVisitor::default();
    visitor.visit_file(&syn::parse_file(&source).unwrap());
    let registered_classes = visitor.classes;
    files_with_extension(source_root, "rs")
        .into_iter()
        .flat_map(|path| {
            let source = fs::read_to_string(&path).unwrap();
            syn::parse_file(&source)
                .unwrap()
                .items
                .into_iter()
                .filter_map(|item| PyClass::from_item(item, &path, source_root))
                .filter(|pyclass| registered_classes.contains(&pyclass.rust_name))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

pub(super) fn files_with_extension(root: &Path, extension: &str) -> Vec<PathBuf> {
    fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter_map(|path| {
            if path.is_dir() {
                Some(files_with_extension(&path, extension))
            } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
                Some(vec![path])
            } else {
                None
            }
        })
        .flatten()
        .collect()
}

impl PyClass {
    fn from_item(item: Item, path: &Path, root: &Path) -> Option<Self> {
        let (name, module, ident, span) = match item {
            Item::Struct(item) => {
                let span = item.span();
                let (name, module) = get_name_and_module_from_attrs(item.attrs)?;
                (name, module, item.ident.clone(), span)
            }
            Item::Enum(item) => {
                let span = item.span();
                let (name, module) = get_name_and_module_from_attrs(item.attrs)?;
                (name, module, item.ident.clone(), span)
            }
            _ => return None,
        };
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

pub(super) fn normalized_path(path: &Path, root: &Path) -> NormalizedPath {
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
