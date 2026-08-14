use std::collections::HashSet;
use std::fs;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use syn::{Expr, Item, Lit, Meta, spanned::Spanned, visit::Visit};
use tap::Pipe;

use crate::paths;

#[derive(Clone, Debug)]
pub(super) struct PyClass {
    pub(super) path: paths::Normalized,
    pub(super) line: usize,
    pub(super) rust_name: String,
    pub(super) python_name: String,
    pub(super) module: Option<String>,
}

impl PyClass {
    fn from_item(item: Item, relative: paths::Relative<'_>) -> Option<Self> {
        let (attrs, ident, span) = get_infos_from_item(item)?;
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

pub(super) fn get_pyclasses(root: &paths::Root, lib_path: &str) -> Vec<PyClass> {
    root.join(lib_path)
        .pipe(fs::read_to_string)
        .unwrap()
        .pipe(RegisteredClassVisitor::visit)
        .pipe(|registered_classes| {
            root.iter_on_extension("rs")
                .flat_map(|path: PathBuf| get_classes_from_file(&path, &registered_classes, root))
                .collect::<Vec<_>>()
        })
}

#[inline]
fn get_classes_from_file(
    path: &Path,
    registered_classes: &HashSet<String>,
    root: &paths::Root,
) -> Vec<PyClass> {
    path.pipe_ref(fs::read_to_string)
        .expect("Failed to read source file")
        .pipe_ref(|source| syn::parse_file(source))
        .expect("Failed to parse source file")
        .items
        .into_iter()
        .filter_map(|item| PyClass::from_item(item, root.make_relative(&path)))
        .filter(|pyclass| registered_classes.contains(&pyclass.rust_name))
        .collect::<Vec<_>>()
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
