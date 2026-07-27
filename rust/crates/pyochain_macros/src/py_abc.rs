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
const SETTER: &str = "setter";
const GETTER: &str = "getter";
const DELETER: &str = "deleter";

// TODO: this code is frankly speaking ugly as fuck, but it works.
// Time-sink to make it pretty but also time-sink when I have issues with it.
// But very useful for many reasons. Will have to be refactored at some point.

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
    BasicMethod,
    /// #[skip]
    Skipped,
    /// #[pyo3(...)]
    /// #[new(...)]
    New(Vec<proc_macro2::TokenStream>),
    /// just #[new]
    NewNoSignature,
    /// #[pyo3(...)] and other attributes
    Signed(Vec<proc_macro2::TokenStream>),
}
fn generate_method(
    trait_ident: &Ident,
    method: &syn::TraitItemFn,
    attrs: Vec<Attribute>,
) -> SynResult<Option<proc_macro2::TokenStream>> {
    let original_ident = &method.sig.ident;
    classify_attrs(attrs)
        .map(|(kind, other_attrs, property_kind)| {
            let python_name = LitStr::new(&original_ident.to_string(), original_ident.span());
            let property = property_kind
                .as_ref()
                .map(|prop| (prop, property_python_name(original_ident, prop)));
            let tokens = match kind {
                AttrKind::Skipped => None,
                AttrKind::BasicMethod => Some(match &property {
                    Some((prop, name)) => quote! { #[#prop(#name)] },
                    None => quote! { #[pyo3(name = #python_name)] },
                }),
                AttrKind::NewNoSignature => Some(quote! {}),
                AttrKind::New(ref tokens) => Some(quote! { #[pyo3(#(#tokens),*)] }),
                AttrKind::Signed(ref tokens) => Some(match &property {
                    Some((prop, name)) => quote! { #[#prop(#name)] #[pyo3(#(#tokens),*)] },
                    None => quote! { #[pyo3(name = #python_name, #(#tokens),*)] },
                }),
            };
            (tokens, other_attrs)
        })
        .and_then(
            |(tokens, other_attrs): (Option<proc_macro2::TokenStream>, Vec<Attribute>)| {
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
            },
        )
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

    signature.inputs.pipe_ref_mut(drop_mut_and_ref_from_pattern);

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
/// Clean-up `mut` and `ref` from wrapper callers.\
/// The generated wrapper only forwards its arguments to the trait method:
///
///     fn py_loc(..., mut pos: usize) {
///         <Self as Trait>::loc(..., pos)
///     }
///
/// Since the wrapper never mutates its parameters,\
/// keeping `mut` (or `ref`) from the original trait signature triggers `unused_mut` warnings after macro expansion.\
/// Strip these qualifiers from the generated signature only.
fn drop_mut_and_ref_from_pattern(inputs: &mut Punctuated<FnArg, Comma>) {
    inputs
        .iter_mut()
        .filter_map(|arg| match arg {
            FnArg::Typed(arg) => Some(arg),
            _ => None,
        })
        .filter_map(|arg| match arg.pat.as_mut() {
            Pat::Ident(pattern) => Some(pattern),
            _ => None,
        })
        .for_each(|pattern| {
            pattern.mutability = None;
            pattern.by_ref = None;
        })
}
fn classify_attrs(attrs: Vec<Attribute>) -> SynResult<(AttrKind, Vec<Attribute>, Option<Ident>)> {
    let mut has_new = false;
    let mut property_kind = None;
    let mut pyo3 = Vec::new();
    let mut other = Vec::new();
    for attr in attrs {
        let path = &attr.path();

        if path.is_ident(SKIP) {
            other.clear();
            return Ok((AttrKind::Skipped, other, None));
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
            } else if path.is_ident(GETTER) || path.is_ident(SETTER) || path.is_ident(DELETER) {
                property_kind = Some(path.get_ident().expect("checked above").clone());
            } else {
                other.push(attr);
            }
        }
    }
    let pyo3_attrs = match (has_new, pyo3.is_empty()) {
        (true, true) => AttrKind::NewNoSignature,
        (false, true) => AttrKind::BasicMethod,
        (true, false) => AttrKind::New(pyo3),
        (false, false) => AttrKind::Signed(pyo3),
    };
    Ok((pyo3_attrs, other, property_kind))
}
fn property_python_name(original_ident: &Ident, prop: &Ident) -> LitStr {
    let name = original_ident.to_string();
    let prefix = if prop.eq(GETTER) {
        "get_"
    } else if prop.eq(SETTER) {
        "set_"
    } else {
        "del_"
    };
    let stripped = name.strip_prefix(prefix).unwrap_or(&name);
    LitStr::new(stripped, original_ident.span())
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
