use quote::{format_ident, quote};
use syn::{
    Attribute, FnArg, Ident, ItemTrait, LitStr, Meta, Pat, TraitItem, Type, parse::ParseStream,
    punctuated::Punctuated, token::Comma,
};
use tap::prelude::*;

use crate::types::SynResult;
const PYO3: &str = "pyo3";
const NEW: &str = "new";
const SKIP: &str = "skip";

pub(crate) fn parse_types(input: ParseStream<'_>) -> SynResult<Punctuated<Type, Comma>> {
    Punctuated::<Type, Comma>::parse_terminated(input)
}

pub(crate) fn generate(
    mut item_trait: ItemTrait,
    types: Punctuated<Type, Comma>,
) -> SynResult<proc_macro2::TokenStream> {
    let methods = item_trait
        .items
        .iter_mut()
        .filter_map(|item| match item {
            TraitItem::Fn(method) => std::mem::take(&mut method.attrs)
                .pipe(|attrs| generate_method(&item_trait.ident, &method, attrs))
                .transpose(),
            _ => None,
        })
        .collect::<SynResult<Vec<_>>>()?;
    let implementations = types.iter().map(|ty| {
        quote! {
            #[pyo3::pymethods]
            impl #ty {
                #(#methods)*
            }
        }
    });

    Ok(quote! {
        #item_trait
        #(#implementations)*
    })
}
enum AttrKind {
    /// no #[pyo3(...)] or #[new(...)] attributes
    Empty,
    /// #[skip]
    Skipped,
    /// #[pyo3(...)]
    /// #[new(...)]
    New(Vec<proc_macro2::TokenStream>),
    /// just #[new]
    NewNoSignature,
    /// #[pyo3(...)] and other attributes
    Other(Vec<proc_macro2::TokenStream>),
}
fn generate_method(
    trait_ident: &Ident,
    method: &syn::TraitItemFn,
    attrs: Vec<Attribute>,
) -> SynResult<Option<proc_macro2::TokenStream>> {
    let original_ident = &method.sig.ident;
    classify_attrs(attrs)
        .map(|(kind, other_attrs)| {
            let python_name = LitStr::new(&original_ident.to_string(), original_ident.span());
            let tokens = match kind {
                AttrKind::Skipped => None,
                AttrKind::Empty => Some(quote! { #[pyo3(name = #python_name)] }),
                AttrKind::NewNoSignature => Some(quote! {}),
                AttrKind::New(ref tokens) => Some(quote! { #[pyo3(#(#tokens),*)] }),
                AttrKind::Other(ref tokens) => {
                    Some(quote! { #[pyo3(name = #python_name, #(#tokens),*)] })
                }
            };
            (tokens, other_attrs)
        })
        .and_then(|(tokens, other_attrs)| {
            tokens
                .map(|pyo3_tokens| {
                    get_quote(
                        method,
                        trait_ident,
                        other_attrs,
                        pyo3_tokens,
                        original_ident,
                    )
                })
                .transpose()
        })
}
fn get_quote(
    method: &syn::TraitItemFn,
    trait_ident: &Ident,
    other_attrs: Vec<Attribute>,
    pyo3_tokens: proc_macro2::TokenStream,
    original_ident: &Ident,
) -> SynResult<proc_macro2::TokenStream> {
    let mut signature = method.sig.clone();
    signature.ident = format_ident!("py_{original_ident}");
    let arguments = get_method_args(method)?;
    let receiver =
        matches!(method.sig.inputs.first(), Some(FnArg::Receiver(_))).then(|| quote! { self, });
    let quote = quote! {
        #(#other_attrs)*
        #pyo3_tokens
        #signature {
            <Self as #trait_ident>::#original_ident(#receiver #(#arguments),*)
        }
    };
    Ok(quote)
}
fn classify_attrs(attrs: Vec<Attribute>) -> SynResult<(AttrKind, Vec<Attribute>)> {
    let mut has_new = false;
    let mut pyo3 = Vec::new();
    let mut other = Vec::new();
    for attr in attrs {
        let path = &attr.path();

        if path.is_ident(SKIP) {
            other.clear();
            return Ok((AttrKind::Skipped, other));
        } else {
            if path.is_ident(NEW) {
                has_new = true;
            }
            if path.is_ident(PYO3) {
                let arg = match attr.meta {
                    Meta::List(list) => Ok(list.tokens),
                    _ => Err(syn::Error::new_spanned(
                        attr,
                        "expected #[pyo3(...)] on a #[py_methods] trait method",
                    )),
                }?;
                pyo3.push(arg);
            } else {
                other.push(attr);
            }
        }
    }
    let pyo3_attrs = match (has_new, pyo3.is_empty()) {
        (true, true) => AttrKind::NewNoSignature,
        (false, true) => AttrKind::Empty,
        (true, false) => AttrKind::New(pyo3),
        (false, false) => AttrKind::Other(pyo3),
    };
    Ok((pyo3_attrs, other))
}

fn get_method_args(method: &syn::TraitItemFn) -> SynResult<Vec<syn::Ident>> {
    method
        .sig
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            FnArg::Receiver(_) => None,
            FnArg::Typed(argument) => Some(match argument.pat.as_ref() {
                Pat::Ident(pattern) => Ok(pattern.ident.clone()),
                pattern => Err(syn::Error::new_spanned(
                    pattern,
                    "#[py_methods] parameters must use identifier patterns",
                )),
            }),
        })
        .collect::<SynResult<Vec<_>>>()
}
