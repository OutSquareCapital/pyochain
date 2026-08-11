use crate::types::{SynResult, TokensVec};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Expr, ExprMatch, Pat};

const INVALID_PATTERN_MSG: &str = "Invalid pattern in try_cast! macro. Use `Type(pat)`, `Type::exact(pat)`, `Some(pat)`, `None`, `(P1, P2, ...)`, an identifier to bind, or `_`";

#[derive(Clone)]
enum Matcher {
    Wild,
    Bind {
        target: TokenStream2,
        name: syn::Ident,
    },
    Cast {
        target: TokenStream2,
        ty: syn::Path,
        exact: bool,
        inner: Box<Matcher>,
    },
    OptionSome {
        target: TokenStream2,
        inner: Box<Matcher>,
    },
    OptionNone {
        target: TokenStream2,
    },
    Tuple {
        target: TokenStream2,
        children: Vec<Matcher>,
    },
}

impl Matcher {
    fn type_name(&self) -> String {
        match self {
            Matcher::Wild => "_".to_string(),
            Matcher::Bind { name, .. } => name.to_string(),
            Matcher::Cast { ty, exact, .. } => {
                let base = quote!(#ty).to_string();
                if *exact {
                    format!("{base}::exact")
                } else {
                    base
                }
            }
            Matcher::OptionSome { inner, .. } => format!("Some({})", inner.type_name()),
            Matcher::OptionNone { .. } => "None".to_string(),
            Matcher::Tuple { children, .. } => format!(
                "({})",
                children
                    .iter()
                    .map(Matcher::type_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    fn wrap(&self, inner: TokenStream2) -> TokenStream2 {
        match self {
            Matcher::Wild => inner,

            Matcher::Bind { target, name } => {
                quote! {
                    let #name = #target;
                    #inner
                }
            }

            Matcher::Cast {
                target,
                ty,
                exact,
                inner: sub,
            } => {
                let val = format_ident!("__cast_val");
                let call = if *exact {
                    quote!((#target).cast_exact::<#ty>())
                } else {
                    quote!((#target).cast::<#ty>())
                };
                let rebased = sub.rebase_target(quote!(#val));
                let body = rebased.wrap(inner);
                quote! {
                    if let ::std::result::Result::Ok(#val) = #call {
                        #body
                    }
                }
            }

            Matcher::OptionSome { target, inner: sub } => {
                let val = format_ident!("__opt_val");
                let rebased = sub.rebase_target(quote!(#val));
                let body = rebased.wrap(inner);
                quote! {
                    if let ::std::option::Option::Some(#val) = &#target {
                        #body
                    }
                }
            }

            Matcher::OptionNone { target } => {
                quote! {
                    if ::std::option::Option::is_none(&#target) {
                        #inner
                    }
                }
            }

            Matcher::Tuple { target, children } => {
                let names: Vec<syn::Ident> = (0..children.len())
                    .map(|i| format_ident!("__tuple_{}", i))
                    .collect();
                let rebased: Vec<Matcher> = children
                    .iter()
                    .zip(&names)
                    .map(|(c, n)| c.rebase_target(quote!(#n)))
                    .collect();
                let body = rebased.iter().rev().fold(inner, |acc, c| c.wrap(acc));
                quote! {
                    let (#(#names),*) = #target;
                    #body
                }
            }
        }
    }

    fn rebase_target(&self, new_target: TokenStream2) -> Matcher {
        match self {
            Matcher::Wild => Matcher::Wild,
            Matcher::Bind { name, .. } => Matcher::Bind {
                target: new_target,
                name: name.clone(),
            },
            Matcher::Cast {
                ty, exact, inner, ..
            } => Matcher::Cast {
                target: new_target,
                ty: ty.clone(),
                exact: *exact,
                inner: inner.clone(),
            },
            Matcher::OptionSome { inner, .. } => Matcher::OptionSome {
                target: new_target,
                inner: inner.clone(),
            },
            Matcher::OptionNone { .. } => Matcher::OptionNone { target: new_target },
            Matcher::Tuple { children, .. } => Matcher::Tuple {
                target: new_target,
                children: children.clone(),
            },
        }
    }
}

pub(crate) fn generate_from_expr(match_expr: ExprMatch) -> TokenStream {
    match_expr
        .arms
        .iter()
        .map(|arm| parse_arm(arm, &match_expr.expr))
        .collect::<SynResult<Vec<Vec<_>>>>()
        .map(|arms| {
            arms.into_iter().flatten().fold(
                (TokensVec::new(), Vec::<String>::new(), None),
                |(mut checks, mut expected_types, fallback), (matcher, body)| {
                    if matches!(matcher, Matcher::Wild) {
                        (checks, expected_types, Some(body))
                    } else {
                        expected_types.push(matcher.type_name());
                        checks.push(matcher.wrap(quote!(return #body;)));
                        (checks, expected_types, fallback)
                    }
                },
            )
        })
        .map(|(checks, expected_types, fallback)| {
            let default_err = get_default_err(fallback, expected_types);
            quote! {
                {
                    #(#checks)*
                    #default_err
                }
            }
        })
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn get_default_err(fallback: Option<TokenStream2>, expected_types: Vec<String>) -> TokenStream2 {
    fallback.unwrap_or_else(|| {
        let err_msg = format!("expected one of: {}", expected_types.join(" | "));
        quote! {
            ::std::result::Result::Err(
                ::pyo3::exceptions::PyTypeError::new_err(#err_msg)
            )
        }
    })
}

fn parse_arm(arm: &syn::Arm, target: &Expr) -> SynResult<Vec<(Matcher, TokenStream2)>> {
    let body = &arm.body;
    let target = match target {
        Expr::Tuple(tuple) => {
            let values = tuple.elems.iter().map(|value| quote!(&#value));
            quote!((#(#values),*))
        }
        _ => quote!(#target),
    };
    match &arm.pat {
        Pat::Wild(_) => Ok(vec![(Matcher::Wild, quote!(#body))]),
        Pat::Or(pat_or) => pat_or
            .cases
            .iter()
            .map(|case| compile_pat(case, target.clone()).map(|m| (m, quote!(#body))))
            .collect(),
        pat => Ok(vec![(compile_pat(pat, target)?, quote!(#body))]),
    }
}

fn compile_pat(pat: &Pat, target: TokenStream2) -> SynResult<Matcher> {
    match pat {
        Pat::Wild(_) => Ok(Matcher::Wild),

        Pat::Paren(pp) => compile_pat(&pp.pat, target),

        Pat::Ident(pi) if pi.subpat.is_none() => Ok(Matcher::Bind {
            target,
            name: pi.ident.clone(),
        }),

        Pat::Path(p) if p.path.is_ident("None") => Ok(Matcher::OptionNone { target }),

        Pat::Tuple(pt) => {
            let children = pt
                .elems
                .iter()
                .map(|p| compile_pat(p, quote!(__pending__)))
                .collect::<SynResult<Vec<_>>>()?;
            Ok(Matcher::Tuple { target, children })
        }

        Pat::TupleStruct(ts) if ts.path.is_ident("Some") => {
            if ts.elems.len() != 1 {
                return Err(syn::Error::new_spanned(
                    ts,
                    "`Some` takes exactly one pattern",
                ));
            }
            let inner = compile_pat(&ts.elems[0], quote!(__pending__))?;
            Ok(Matcher::OptionSome {
                target,
                inner: Box::new(inner),
            })
        }

        Pat::TupleStruct(ts) => {
            if ts.elems.len() != 1 {
                return Err(syn::Error::new_spanned(
                    ts,
                    "Type cast pattern takes exactly one inner pattern, e.g. `Type(x)` or `Type(_)`",
                ));
            }
            let (ty, exact) = strip_exact_path(&ts.path);
            let inner = compile_pat(&ts.elems[0], quote!(__pending__))?;
            Ok(Matcher::Cast {
                target,
                ty,
                exact,
                inner: Box::new(inner),
            })
        }

        _ => Err(syn::Error::new_spanned(pat, INVALID_PATTERN_MSG)),
    }
}

fn strip_exact_path(path: &syn::Path) -> (syn::Path, bool) {
    let mut segments = path.segments.clone();
    let exact = segments.len() >= 2 && segments.last().map(|s| s.ident == "exact").unwrap_or(false);
    if exact {
        segments.pop();
        segments.pop_punct();
    }
    let mut p = path.clone();
    p.segments = segments;
    (p, exact)
}
