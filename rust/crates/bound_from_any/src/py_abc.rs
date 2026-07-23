use quote::{format_ident, quote};
use syn::{
    Attribute, FnArg, Ident, ItemTrait, LitStr, Meta, Pat, TraitItem, Type, parse::ParseStream,
    punctuated::Punctuated, token::Comma,
};

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
    let methods = get_methods(&mut item_trait.items, &item_trait.ident)?;
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
fn get_methods(
    items: &mut Vec<TraitItem>,
    trait_ident: &Ident,
) -> SynResult<Vec<proc_macro2::TokenStream>> {
    items
        .iter_mut()
        .filter_map(|item| match item {
            TraitItem::Fn(method) => Some(method),
            _ => None,
        })
        .filter_map(|method| export_method(&mut method.attrs).map(|attrs| (method, attrs)))
        .map(|(method, attrs)| generate_method(&trait_ident, method, attrs))
        .collect::<SynResult<Vec<_>>>()
}
fn export_method(attrs: &mut Vec<Attribute>) -> Option<Vec<Attribute>> {
    attrs
        .iter()
        .position(|attr| attr.path().is_ident(SKIP))
        .map(|position| {
            attrs.remove(position);
            None
        })
        .unwrap_or_else(|| Some(std::mem::take(attrs)))
}
fn generate_method(
    trait_ident: &Ident,
    method: &syn::TraitItemFn,
    attrs: Vec<Attribute>,
) -> SynResult<proc_macro2::TokenStream> {
    let original_ident = &method.sig.ident;
    let wrapper_ident = format_ident!("py_{original_ident}");
    let python_name = LitStr::new(&original_ident.to_string(), original_ident.span());
    let mut signature = method.sig.clone();
    signature.ident = wrapper_ident;
    let arguments = method
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
        .collect::<SynResult<Vec<_>>>()?;
    let is_constructor = attrs.iter().any(|attr| attr.path().is_ident(NEW));
    let (pyo3_args, attrs) =
        attrs
            .into_iter()
            .try_fold((Vec::new(), Vec::new()), |(mut pyo3, mut other), attr| {
                if attr.path().is_ident(PYO3) {
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
                Ok::<(Vec<proc_macro2::TokenStream>, Vec<Attribute>), syn::Error>((pyo3, other))
            })?;
    let pyo3_attr = match (is_constructor, pyo3_args.is_empty()) {
        (true, true) => quote! {},
        (true, false) => quote! { #[pyo3(#(#pyo3_args),*)] },
        (false, _) => quote! { #[pyo3(name = #python_name, #(#pyo3_args),*)] },
    };
    let receiver =
        matches!(method.sig.inputs.first(), Some(FnArg::Receiver(_))).then(|| quote! { self, });

    Ok(quote! {
        #(#attrs)*
        #pyo3_attr
        #signature {
            <Self as #trait_ident>::#original_ident(#receiver #(#arguments),*)
        }
    })
}
