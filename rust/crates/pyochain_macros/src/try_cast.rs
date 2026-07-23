use crate::types::{SynResult, TokensVec};
use proc_macro::TokenStream;
use quote::quote;
use syn::{ExprMatch, Pat};

const INVALID_PATTERN_MSG: &str =
    "Invalid pattern in try_cast! macro. Use `Type`, `var: Type`, `T1 | T2`, `(T1, T2)` or `_`";

enum ParsedArm {
    Check {
        check: proc_macro2::TokenStream,
        ty_name: String,
    },
    Fallback(proc_macro2::TokenStream),
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
                |(mut checks, mut expected_types, fallback), arm| match arm {
                    ParsedArm::Check { check, ty_name } => {
                        checks.push(check);
                        expected_types.push(ty_name);
                        (checks, expected_types, fallback)
                    }
                    ParsedArm::Fallback(body) => (checks, expected_types, Some(quote!(#body))),
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

fn get_default_err(
    fallback: Option<proc_macro2::TokenStream>,
    expected_types: Vec<String>,
) -> proc_macro2::TokenStream {
    fallback.unwrap_or_else(|| {
        let err_msg = format!("expected one of: {}", expected_types.join(" | "));
        quote! {
            ::std::result::Result::Err(
                ::pyo3::exceptions::PyTypeError::new_err(#err_msg)
            )
        }
    })
}

fn parse_arm(arm: &syn::Arm, target: &syn::Expr) -> SynResult<Vec<ParsedArm>> {
    let body = &(*arm).body;

    match &arm.pat {
        Pat::Wild(_) => Ok(vec![ParsedArm::Fallback(quote!(#body))]),

        Pat::Or(pat_or) => pat_or
            .cases
            .iter()
            .map(|case| parse_single_pat(case, target, body))
            .collect(),

        pat => parse_single_pat(pat, target, body).map(|check| vec![check]),
    }
}

fn parse_single_pat(pat: &syn::Pat, target: &syn::Expr, body: &syn::Expr) -> SynResult<ParsedArm> {
    match pat {
        Pat::Tuple(pat_tuple) => match target {
            syn::Expr::Tuple(expr_tuple) => (pat_tuple.elems.len() == expr_tuple.elems.len())
                .then(|| {
                    let mut current_check = quote!(return #body;);
                    let type_names = pat_tuple
                        .elems
                        .iter()
                        .zip(expr_tuple.elems.iter())
                        .rev()
                        .map(|(p, t)| {
                            extract_cast_info(p, t).map(|(var, ty, ty_str)| {
                                current_check = quote! {
                                    if let ::std::result::Result::Ok(#var) = #t.cast::<#ty>() {
                                        #current_check
                                    }
                                };

                                ty_str
                            })
                        })
                        .rev()
                        .collect::<SynResult<Vec<_>>>()?
                        .join(", ");
                    let ty_name = format!("({})", type_names);

                    Ok(ParsedArm::Check {
                        check: current_check,
                        ty_name,
                    })
                })
                .ok_or_else(|| {
                    syn::Error::new_spanned(
                        pat_tuple,
                        "Incorrect number of elements in tuple pattern",
                    )
                })?,
            _ => Err(syn::Error::new_spanned(
                target,
                "Tuple pattern does not match a tuple target",
            )),
        },

        // Motifs simples (Ident, Path, Type, Paren)
        _ => {
            let (var, ty, ty_str) = extract_cast_info(pat, target)?;
            Ok(ParsedArm::Check {
                check: quote! {
                    if let ::std::result::Result::Ok(#var) = #target.cast::<#ty>() {
                        return #body;
                    }
                },
                ty_name: ty_str,
            })
        }
    }
}

fn extract_cast_info(
    pat: &syn::Pat,
    target: &syn::Expr,
) -> SynResult<(proc_macro2::TokenStream, proc_macro2::TokenStream, String)> {
    match pat {
        Pat::Ident(pat_ident) => {
            let ty = &pat_ident.ident;
            Ok((quote!(#target), quote!(#ty), quote!(#ty).to_string()))
        }
        Pat::Path(pat_path) => Ok((
            quote!(#target),
            quote!(#pat_path),
            quote!(#pat_path).to_string(),
        )),
        Pat::Type(pat_type) => {
            let var = &pat_type.pat;
            let ty = &pat_type.ty;
            Ok((quote!(#var), quote!(#ty), quote!(#ty).to_string()))
        }
        Pat::Paren(pat_paren) => extract_cast_info(&pat_paren.pat, target),
        _ => Err(syn::Error::new_spanned(pat, INVALID_PATTERN_MSG)),
    }
}
