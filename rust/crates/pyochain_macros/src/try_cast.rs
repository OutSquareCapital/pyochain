use crate::types::SynResult;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Arm, ExprMatch, Ident, Pat, PatIdent, PatTuple, PatTupleStruct, Path, punctuated::Punctuated,
};
use tap::prelude::*;
const INVALID_PATTERN_MSG: &str = "use `Case::Type(binding)` or `CaseExact::Type(binding)`";
const CASE: &str = "Case";
const CASE_EXACT: &str = "CaseExact";
/// Main enum for the two variants of `try_cast` macro: borrowed and owned.
pub(super) enum Cast {
    Borrowed,
    Owned,
}
/// Generate `Case` and `CaseExact` bindings for a given pattern, returning the rewritten pattern and a list of cases.
struct Case {
    binding: Ident,
    output: Ident,
    ty: Path,
    exact: bool,
}
impl Cast {
    pub(super) fn generate(self, match_expr: ExprMatch) -> TokenStream {
        run_pipeline(match_expr, self)
            .unwrap_or_else(syn::Error::into_compile_error)
            .into()
    }
}

fn run_pipeline(match_expr: ExprMatch, mode: Cast) -> SynResult<TokenStream2> {
    let subject = &match_expr.expr;
    match_expr
        .arms
        .iter()
        .map(|arm| generate_match_arm(arm, &mode))
        .collect::<SynResult<Vec<Vec<_>>>>()?
        .into_iter()
        .flatten()
        .pipe(|arms| quote!(match #subject { #(#arms)* }))
        .pipe(Ok)
}

fn generate_match_arm(arm: &Arm, mode: &Cast) -> SynResult<Vec<TokenStream2>> {
    match &arm.pat {
        Pat::Or(pattern) => {
            let rewritten = pattern
                .cases
                .iter()
                .map(rewrite_pattern)
                .collect::<SynResult<Vec<_>>>()?;
            if rewritten.iter().all(|(_, cases)| cases.is_empty()) {
                generate_match_pattern(&arm.pat, &arm.body, mode).map(|arm| vec![arm])
            } else {
                rewritten
                    .iter()
                    .map(|(pattern, cases)| generate_rewritten_arm(pattern, cases, &arm.body, mode))
                    .collect()
            }
        }
        pattern => generate_match_pattern(pattern, &arm.body, mode).map(|arm| vec![arm]),
    }
}

fn generate_match_pattern(pattern: &Pat, body: &syn::Expr, mode: &Cast) -> SynResult<TokenStream2> {
    let (pattern, cases) = rewrite_pattern(pattern)?;
    generate_rewritten_arm(&pattern, &cases, body, mode)
}

fn generate_rewritten_arm(
    pattern: &TokenStream2,
    cases: &[Case],
    body: &syn::Expr,
    mode: &Cast,
) -> SynResult<TokenStream2> {
    let casts = cases.iter().map(|case| mode.new_arm(case));
    let checks = cases.iter().map(|case| {
        let binding = &case.binding;
        let ty = &case.ty;
        if case.exact {
            quote!(#binding.is_exact_instance_of::<#ty>())
        } else {
            quote!(#binding.is_instance_of::<#ty>())
        }
    });

    match checks.len() {
        0 => Ok(quote!(#pattern => #body,)),
        _ => Ok(quote!(#pattern if #(#checks)&&* => { #(#casts)* #body },)),
    }
}

impl Cast {
    fn new_arm(&self, case: &Case) -> TokenStream2 {
        let binding = &case.binding;
        let output = &case.output;
        let ty = &case.ty;
        match self {
            Self::Borrowed => quote! {
                // SAFETY: this arm's guard proves the concrete Python type.
                let #output = unsafe { #binding.cast_unchecked::<#ty>() };
            },
            Self::Owned => quote! {
                // SAFETY: this arm's guard proves the concrete Python type.
                let #output = unsafe { #binding.cast_into_unchecked::<#ty>() };
            },
        }
    }
}

fn rewrite_tuple(tuple: &PatTuple) -> SynResult<(TokenStream2, Vec<Case>)> {
    let rewritten = tuple
        .elems
        .iter()
        .map(rewrite_pattern)
        .collect::<SynResult<Vec<_>>>()?;
    let (patterns, cases): (Vec<_>, Vec<_>) = rewritten.into_iter().unzip();
    let cases = cases.into_iter().flatten().collect::<Vec<_>>();
    Ok((quote!((#(#patterns),*)), cases))
}

fn rewrite_pattern(pattern: &Pat) -> SynResult<(TokenStream2, Vec<Case>)> {
    match pattern {
        Pat::Tuple(tuple) => rewrite_tuple(tuple),
        Pat::TupleStruct(tuple_struct) => rewrite_case(tuple_struct),
        _ => Ok((quote!(#pattern), Vec::new())),
    }
}

fn rewrite_case(pattern: &PatTupleStruct) -> SynResult<(TokenStream2, Vec<Case>)> {
    let Some(marker) = pattern.path.segments.first() else {
        return Err(syn::Error::new_spanned(pattern, INVALID_PATTERN_MSG));
    };
    let exact = match marker.ident.to_string().as_str() {
        CASE => false,
        CASE_EXACT => true,
        _ => return Ok((quote!(#pattern), Vec::new())),
    };
    if pattern.path.segments.len() != 2 || pattern.elems.len() != 1 {
        return Err(syn::Error::new_spanned(pattern, INVALID_PATTERN_MSG));
    }
    let ty = Path {
        leading_colon: None,
        segments: Punctuated::from_iter([pattern.path.segments[1].clone()]),
    };
    let Pat::Ident(PatIdent {
        ident,
        subpat: None,
        ..
    }) = &pattern.elems[0]
    else {
        return Err(syn::Error::new_spanned(pattern, INVALID_PATTERN_MSG));
    };
    let binding = format_ident!("__try_cast_{}", ident);
    Ok((
        quote!(#binding),
        vec![Case {
            binding,
            output: ident.clone(),
            ty,
            exact,
        }],
    ))
}
