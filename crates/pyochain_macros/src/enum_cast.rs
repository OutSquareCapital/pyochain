use crate::types::{SynResult, TokensVec};
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Attribute, Data, DeriveInput, Fields, GenericArgument, Ident, LitStr, PathArguments, Type,
    Variant,
};
pub(crate) fn generate_from_input(input: DeriveInput) -> TokenStream {
    match &input.data {
        Data::Enum(data_enum) => data_enum
            .variants
            .iter()
            .map(get_arm_and_name)
            .collect::<SynResult<(TokensVec, Vec<String>)>>()
            .map(|(arms, names)| gen_impl(input, &arms, &names)),
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "BoundFromAny only supports enums",
        )),
    }
    .unwrap_or_else(syn::Error::into_compile_error)
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

fn get_arm_and_name(variant: &Variant) -> SynResult<(proc_macro2::TokenStream, String)> {
    let field = match &variant.fields {
        Fields::Unnamed(f) if f.unnamed.len() == 1 => Ok(f.unnamed.iter().next().unwrap()),
        _ => Err(syn::Error::new_spanned(
            variant,
            "variants must contain exactly one field",
        )),
    }?;

    let ty = &field.ty;
    let mode = Mode::new(&variant.attrs);

    let inner = match mode {
        Mode::Extract => ty,
        Mode::Cast | Mode::CastExact => bound_inner(ty)?,
    };

    let name = match inner {
        Type::Path(p) => p.path.segments.last().unwrap().ident.to_string(),
        _ => quote!(#inner).to_string(),
    };
    let arm = mode.gen_arm(&variant.ident, inner);

    Ok((arm, name))
}
#[derive(Copy, Clone)]
enum Mode {
    Cast,
    CastExact,
    Extract,
}

impl Mode {
    fn new(attrs: &[Attribute]) -> Self {
        attrs
            .iter()
            .map(Attribute::path)
            .find_map(|a| {
                if a.is_ident("cast_exact") {
                    Some(Self::CastExact)
                } else if a.is_ident("extract") {
                    Some(Self::Extract)
                } else {
                    None
                }
            })
            .unwrap_or(Self::Cast)
    }

    fn gen_arm(self, ident: &Ident, inner: &Type) -> proc_macro2::TokenStream {
        match self {
            Self::Cast => quote! {
                if obj.is_instance_of::<#inner>() {
                    return Ok(Self::#ident(unsafe {obj.to_owned().cast_into_unchecked::<#inner>()}));
                }
            },
            Self::CastExact => quote! {
                if obj.is_exact_instance_of::<#inner>() {
                    return Ok(Self::#ident(unsafe {obj.to_owned().cast_into_unchecked::<#inner>()}));
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

fn bound_inner(ty: &Type) -> SynResult<&Type> {
    match ty {
        Type::Path(tp) => tp
            .path
            .segments
            .last()
            .filter(|seg| seg.ident == "Bound")
            .and_then(|seg| match &seg.arguments {
                PathArguments::AngleBracketed(args) => args
                    .args
                    .iter()
                    .filter_map(|a| match a {
                        GenericArgument::Type(t) => Some(t),
                        _ => None,
                    })
                    .next_back(),
                _ => None,
            })
            .ok_or_else(|| expected_bound_err(ty)),
        _ => Err(expected_bound_err(ty)),
    }
}
fn expected_bound_err(ty: &Type) -> syn::Error {
    syn::Error::new_spanned(ty, "expected Bound<'py, T>")
}

#[cfg(test)]
mod tests {
    use super::get_arm_and_name;
    use syn::{Variant, parse_quote};

    #[test]
    fn extract_attribute_on_variant_is_honored() {
        let variant: Variant = parse_quote!(#[extract] Usize(usize));
        let (arm, name) = get_arm_and_name(&variant).unwrap();
        assert_eq!(name, "usize");
        assert!(arm.to_string().contains("extract"));
    }

    #[test]
    fn bound_variant_defaults_to_cast() {
        let variant: Variant = parse_quote!(PySlice(Bound<'py, PySlice>));
        let (arm, name) = get_arm_and_name(&variant).unwrap();
        assert_eq!(name, "PySlice");
        assert!(arm.to_string().contains("is_instance_of"));
    }

    #[test]
    fn cast_exact_attribute_on_variant_is_honored() {
        let variant: Variant = parse_quote!(#[cast_exact] List(Bound<'py, PyList>));
        let (arm, name) = get_arm_and_name(&variant).unwrap();
        assert_eq!(name, "PyList");
        assert!(arm.to_string().contains("is_exact_instance_of"));
    }

    #[test]
    fn non_bound_type_without_extract_is_rejected() {
        let variant: Variant = parse_quote!(Usize(usize));
        assert!(get_arm_and_name(&variant).is_err());
    }
}
