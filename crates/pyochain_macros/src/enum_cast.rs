use crate::types::{SynResult, TokensVec};
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Attribute, Data, DeriveInput, Fields, GenericArgument, Ident, LitStr, PathArguments, Type,
    punctuated::Punctuated, token,
};
pub(crate) fn generate_from_input(input: DeriveInput) -> TokenStream {
    get_variants(&input)
        .and_then(|variants| get_arms_and_names(&variants))
        .map_or_else(syn::Error::into_compile_error, |(arms, names)| {
            gen_impl(input, &arms, names.as_slice())
        })
        .into()
}

fn gen_impl(input: DeriveInput, arms: &TokensVec, names: &[String]) -> proc_macro2::TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let ident = input.ident;

    let expected = LitStr::new(
        &format!("expected one of: {}", names.join(" | ")),
        proc_macro2::Span::call_site(),
    );

    quote! {
        impl #impl_generics ::pyo3::conversion::FromPyObject<'_, 'py>
            for #ident #ty_generics
        #where_clause
        {
            type Error = ::pyo3::PyErr;

            #[inline]
            fn extract(
                obj: ::pyo3::Borrowed<'_, 'py, ::pyo3::PyAny>,
            ) -> ::pyo3::PyResult<Self> {

                #(#arms)*

                Err(::pyo3::exceptions::PyTypeError::new_err(#expected))
            }
        }
    }
}

fn get_arms_and_names(
    variants: &Punctuated<syn::Variant, token::Comma>,
) -> SynResult<(TokensVec, Vec<String>)> {
    variants
        .iter()
        .map(|variant| {
            let field = match &variant.fields {
                Fields::Unnamed(f) if f.unnamed.len() == 1 => Ok(f.unnamed.iter().next().unwrap()),
                _ => Err(syn::Error::new_spanned(
                    variant,
                    "variants must contain exactly one field",
                )),
            }?;

            let ty = &field.ty;
            let inner = bound_inner(ty)?;
            let name = match inner {
                Type::Path(p) => p.path.segments.last().unwrap().ident.to_string(),
                _ => quote!(#inner).to_string(),
            };
            let arm = Mode::new(&field.attrs).gen_arm(&variant.ident, inner);

            Ok((arm, name))
        })
        .collect()
}

#[derive(Copy, Clone)]
enum Mode {
    Cast,
    CastExact,
    Extract,
}

impl Mode {
    fn new(attrs: &[Attribute]) -> Self {
        let has = |name| attrs.iter().any(|a| a.path().is_ident(name));

        if has("cast_exact") {
            Self::CastExact
        } else if has("extract") {
            Self::Extract
        } else {
            Self::Cast
        }
    }

    fn gen_arm(self, ident: &Ident, inner: &Type) -> proc_macro2::TokenStream {
        match self {
            Self::Cast => quote! {
                if let Ok(v) = obj.cast::<#inner>() {
                    return Ok(Self::#ident(v.to_owned()));
                }
            },
            Self::CastExact => quote! {
                if let Ok(v) = obj.cast_exact::<#inner>() {
                    return Ok(Self::#ident(v.to_owned()));
                }
            },
            Self::Extract => quote! {
                if let Ok(v) = obj.extract::<#inner>() {
                    return Ok(Self::#ident(v));
                }
            },
        }
    }
}
fn get_variants(input: &DeriveInput) -> SynResult<Punctuated<syn::Variant, token::Comma>> {
    match &input.data {
        Data::Enum(data_enum) => Ok(data_enum.variants.clone()),
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "BoundFromAny only supports enums",
        )),
    }
}

fn bound_inner(ty: &Type) -> SynResult<&Type> {
    match ty {
        Type::Path(tp) => tp
            .path
            .segments
            .last()
            .filter(|seg| seg.ident == "Bound") // Fix: seg.ident == "Bound"
            .and_then(|seg| match &seg.arguments {
                PathArguments::AngleBracketed(args) => args
                    .args
                    .iter()
                    .filter_map(|a| match a {
                        GenericArgument::Type(t) => Some(t),
                        _ => None,
                    })
                    .last(),
                _ => None,
            })
            .ok_or_else(|| expected_bound_err(ty)),
        _ => Err(expected_bound_err(ty)),
    }
}
fn expected_bound_err(ty: &Type) -> syn::Error {
    syn::Error::new_spanned(ty, "expected Bound<'py, T>")
}
