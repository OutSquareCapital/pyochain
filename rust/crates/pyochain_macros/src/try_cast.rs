use crate::types::SynResult;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Arm, ExprMatch, Pat, PatIdent, PatTuple, PatTupleStruct, Path, punctuated::Punctuated};
use tap::prelude::*;
const INVALID_PATTERN_MSG: &str = "use `Case::Type(binding)` or `CaseExact::Type(binding)`";
const CASE: &str = "Case";
const CASE_EXACT: &str = "CaseExact";
pub(crate) fn generate_from_expr(match_expr: ExprMatch) -> TokenStream {
    run_pipeline(match_expr)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn run_pipeline(match_expr: ExprMatch) -> SynResult<TokenStream2> {
    let subject = &match_expr.expr;
    match_expr
        .arms
        .iter()
        .map(generate_match_arm)
        .collect::<SynResult<Vec<Vec<_>>>>()?
        .into_iter()
        .flatten()
        .pipe(|arms| quote!(match #subject { #(#arms)* }))
        .pipe(Ok)
}

fn generate_match_arm(arm: &Arm) -> SynResult<Vec<TokenStream2>> {
    match &arm.pat {
        Pat::Or(pattern) => {
            let rewritten = pattern
                .cases
                .iter()
                .map(rewrite_pattern)
                .collect::<SynResult<Vec<_>>>()?;
            if rewritten.iter().all(|(_, cases)| cases.is_empty()) {
                generate_match_pattern(&arm.pat, &arm.body).map(|arm| vec![arm])
            } else {
                rewritten
                    .iter()
                    .map(|(pattern, cases)| generate_rewritten_arm(pattern, cases, &arm.body))
                    .collect()
            }
        }
        pattern => generate_match_pattern(pattern, &arm.body).map(|arm| vec![arm]),
    }
}

fn generate_match_pattern(pattern: &Pat, body: &syn::Expr) -> SynResult<TokenStream2> {
    let (pattern, cases) = rewrite_pattern(pattern)?;
    generate_rewritten_arm(&pattern, &cases, body)
}

fn generate_rewritten_arm(
    pattern: &TokenStream2,
    cases: &[Case],
    body: &syn::Expr,
) -> SynResult<TokenStream2> {
    let casts = cases.iter().map(|case| {
        let binding = &case.binding;
        let output = &case.output;
        let ty = &case.ty;
        quote! {
            // SAFETY: this arm's guard proves the concrete Python type.
            let #output = unsafe { #binding.cast_unchecked::<#ty>() };
        }
    });
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

struct Case {
    binding: syn::Ident,
    output: syn::Ident,
    ty: Path,
    exact: bool,
}
