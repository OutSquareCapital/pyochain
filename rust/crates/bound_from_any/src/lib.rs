use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Attribute, Data, DataEnum, DeriveInput, Fields, GenericArgument, Ident, LitStr, PathArguments,
    Type, parse_macro_input,
};

///`BoundFromAny` is a derive macro that generates a `FromPyObject` implementation for enums containing `Bound<'py, T>` variants.\
/// It allows matching a Python object against multiple possible PyO3 types using `cast` or `cast_exact`.\
/// The generated extractor tries variants in declaration order.\
/// The benefit vs deriving `FromPyObject` is that the actual type check is much more efficient and lightweight, since the scope is more narrow, and specialized for `Bound<'py, T>`.\
/// On tests, with 3 match arms, 2 success and one PyErr, the output assembly is the same as a hand-written implementation and try_cast! macro calls.\
/// **Note**: The claims of efficiency are verified by ASM cross-check with Gemini and GPT, I'm not an expert on this, so take it with a grain of salt).
///## Example
///
///```rust
///#[derive(BoundFromAny)]
///enum Container<'py> {
///    #[cast]
///    List(Bound<'py, PyList>),
///    #[cast_exact]
///    Tuple(Bound<'py, PyTuple>),
///}
/// fn process(obj: Bound<'_, PyAny>) -> PyResult<()> {
///    match obj.extract::<Container>()? {
///        Container::List(list) => {
///            println!("got a list");
///        }
///        Container::Tuple(tuple) => {
///            println!("got a tuple");
///        }
///    }
///
///    Ok(())
///}
///```
#[proc_macro_derive(BoundFromAny, attributes(cast, cast_exact, extract))]
pub fn derive_bound_from_any(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let ident = input.ident;
    let generics = input.generics;

    let Data::Enum(DataEnum { variants, .. }) = input.data else {
        panic!("BoundFromAny only supports enums");
    };

    let mut arms = Vec::new();
    let mut names = Vec::new();

    for variant in variants {
        let vident = variant.ident;

        let field = match variant.fields {
            Fields::Unnamed(f) if f.unnamed.len() == 1 => f.unnamed.into_iter().next().unwrap(),
            _ => panic!("variants must contain exactly one field"),
        };

        let ty = field.ty;

        names.push(display_name(&ty));

        arms.push(gen_arm(&vident, &ty, mode(&field.attrs)));
    }

    let expected = LitStr::new(
        &format!("expected one of: {}", names.join(" | ")),
        proc_macro2::Span::call_site(),
    );

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

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
    .into()
}

#[derive(Copy, Clone)]
enum Mode {
    Cast,
    CastExact,
}

fn has(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|a| a.path().is_ident(name))
}

fn mode(attrs: &[Attribute]) -> Mode {
    if has(attrs, "cast") {
        Mode::Cast
    } else if has(attrs, "cast_exact") {
        Mode::CastExact
    } else {
        Mode::Cast
    }
}
fn gen_arm(ident: &Ident, ty: &Type, mode: Mode) -> proc_macro2::TokenStream {
    let Some(inner) = bound_inner(ty) else {
        panic!("BoundFromAny only supports Bound<T>");
    };

    match mode {
        Mode::Cast => {
            quote! {
                if let Ok(v) = obj.cast::<#inner>() {
                    return Ok(Self::#ident(v.to_owned()));
                }
            }
        }

        Mode::CastExact => {
            quote! {
                if let Ok(v) = obj.cast_exact::<#inner>() {
                    return Ok(Self::#ident(v.to_owned()));
                }
            }
        }
    }
}
fn display_name(ty: &Type) -> String {
    match &bound_inner(ty).expect("expected Bound<T>") {
        Type::Path(p) => p.path.segments.last().unwrap().ident.to_string(),
        _ => quote!(#ty).to_string(),
    }
}
fn bound_inner(ty: &Type) -> Option<Type> {
    let Type::Path(tp) = ty else {
        return None;
    };

    let seg = tp.path.segments.last()?;

    if seg.ident != "Bound" {
        return None;
    }

    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };

    args.args
        .iter()
        .filter_map(|a| match a {
            GenericArgument::Type(t) => Some(t.clone()),
            _ => None,
        })
        .last()
}
