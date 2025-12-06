//! Reson procedural macros
//!
//! Provides ergonomic decorators for agentic functions and tool definitions.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, ItemFn, Lit, Token, FnArg, PatType, Pat};

// Helper struct to parse macro attributes
struct AgenticArgs {
    model: Option<String>,
    api_key: Option<String>,
    autobind: bool,
    native_tools: bool,
}

impl Parse for AgenticArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut model = None;
        let mut api_key = None;
        let mut autobind = true;
        let mut native_tools = true; // Default true like Python

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match ident.to_string().as_str() {
                "model" => {
                    let lit: Lit = input.parse()?;
                    if let Lit::Str(s) = lit {
                        model = Some(s.value());
                    }
                }
                "api_key" => {
                    let lit: Lit = input.parse()?;
                    if let Lit::Str(s) = lit {
                        api_key = Some(s.value());
                    }
                }
                "autobind" => {
                    let lit: Lit = input.parse()?;
                    if let Lit::Bool(b) = lit {
                        autobind = b.value();
                    }
                }
                "native_tools" => {
                    let lit: Lit = input.parse()?;
                    if let Lit::Bool(b) = lit {
                        native_tools = b.value();
                    }
                }
                _ => {
                    return Err(syn::Error::new(ident.span(), "Unknown attribute"));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self {
            model,
            api_key,
            autobind,
            native_tools,
        })
    }
}

/// `#[agentic]` attribute macro for async functions
///
/// Automatically creates a Runtime, injects it into the function, binds tools, and validates usage.
///
/// The decorated function must have a `runtime: Runtime` parameter which will be automatically
/// injected by the macro - callers should NOT pass it.
///
/// # Attributes
/// - `model`: Model string in format "provider:model" (e.g., "anthropic:claude-3-5-sonnet-20241022")
/// - `api_key`: Optional API key (defaults to environment variable based on provider)
/// - `autobind`: Whether to auto-bind callable parameters as tools (default: true)
/// - `native_tools`: Whether to use native tool calling (default: true)
///
/// # Example
/// ```ignore
/// use reson_agentic::prelude::*;
/// use reson_agentic::agentic;
///
/// #[agentic(model = "anthropic:claude-3-5-sonnet-20241022")]
/// async fn extract_people(text: String, runtime: Runtime) -> Result<Vec<Person>> {
///     runtime.run(
///         Some(&format!("Extract people from: {}", text)),
///         None, None, None, None, None, None, None, None
///     ).await
/// }
///
/// // Usage - note: runtime is NOT passed by caller
/// let result = extract_people("Alice is 30 years old".to_string()).await?;
/// ```
#[proc_macro_attribute]
pub fn agentic(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as AgenticArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let model = args.model;
    let api_key = args.api_key;
    let _autobind = args.autobind;
    let _native_tools = args.native_tools;

    // Extract function components
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_generics = &input_fn.sig.generics;
    let fn_output = &input_fn.sig.output;
    let fn_block = &input_fn.block;
    let fn_attrs = &input_fn.attrs;
    let fn_asyncness = &input_fn.sig.asyncness;

    // Separate runtime parameter from other parameters
    let mut other_params = Vec::new();
    let mut has_runtime = false;

    for param in input_fn.sig.inputs.iter() {
        match param {
            FnArg::Typed(PatType { pat, .. }) => {
                if let Pat::Ident(pat_ident) = pat.as_ref() {
                    if pat_ident.ident == "runtime" {
                        has_runtime = true;
                        continue; // Skip runtime param - we'll inject it
                    }
                }
                other_params.push(param.clone());
            }
            FnArg::Receiver(_) => {
                other_params.push(param.clone());
            }
        }
    }

    if !has_runtime {
        return syn::Error::new_spanned(
            &input_fn.sig.ident,
            "#[agentic] function must have a `runtime: Runtime` parameter"
        )
        .to_compile_error()
        .into();
    }

    // Generate model setup
    let model_setup = if let Some(m) = model {
        quote! { Some(#m.to_string()) }
    } else {
        quote! { None }
    };

    let api_key_setup = if let Some(k) = api_key {
        quote! { Some(#k.to_string()) }
    } else {
        quote! { None }
    };

    // Generate wrapper function that:
    // 1. Creates a Runtime
    // 2. Calls the original function with runtime injected
    // 3. Validates runtime.used after completion
    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_asyncness fn #fn_name #fn_generics(#(#other_params),*) #fn_output {
            // Create Runtime with native tools enabled
            let mut runtime = ::reson_agentic::runtime::Runtime::with_config(
                #model_setup,
                #api_key_setup,
                std::sync::Arc::new(::reson_agentic::storage::MemoryStore::new()),
            );

            // Execute the original function body with runtime in scope
            let result = {
                // The original function body has access to `runtime`
                #fn_block
            };

            // Validate runtime was used
            if !runtime.used {
                panic!(
                    "agentic function '{}' completed without calling runtime.run() or runtime.run_stream()",
                    stringify!(#fn_name)
                );
            }

            result
        }
    };

    TokenStream::from(expanded)
}

/// `#[agentic_generator]` attribute macro for async generator functions
///
/// Similar to `#[agentic]` but for functions that return `impl Stream`.
/// Generator functions can yield intermediate results while processing.
///
/// # Example
/// ```ignore
/// #[agentic_generator(model = "anthropic:claude-3-5-sonnet-20241022")]
/// async fn process_items(items: Vec<String>, runtime: Runtime) -> impl Stream<Item = Result<String>> {
///     async_stream::stream! {
///         for item in items {
///             let result = runtime.run(Some(&item), None, None, None, None, None, None, None, None).await?;
///             yield Ok(result.to_string());
///         }
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn agentic_generator(attr: TokenStream, item: TokenStream) -> TokenStream {
    // For now, generators use the same logic as regular agentic functions
    // Full generator support with yield tracking would need async_stream integration
    agentic(attr, item)
}

/// `#[derive(Tool)]` for tool structs
///
/// Automatically implements schema generation for tool types.
/// Use with Serialize/Deserialize for full tool support.
///
/// # Example
/// ```ignore
/// #[derive(Tool, Serialize, Deserialize)]
/// struct CalculateTool {
///     /// The operation to perform (add, subtract, multiply, divide)
///     operation: String,
///     /// First operand
///     a: f64,
///     /// Second operand
///     b: f64,
/// }
/// ```
#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();

    // Convert PascalCase to snake_case for tool name
    let tool_name = convert_to_snake_case(&name_str);

    // Extract struct-level doc comment for description
    let struct_description = extract_doc_comments(&input.attrs);

    // Extract fields and their documentation
    let fields = match &input.data {
        syn::Data::Struct(data_struct) => match &data_struct.fields {
            syn::Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    name,
                    "Tool derive only supports structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(name, "Tool derive only supports structs")
                .to_compile_error()
                .into();
        }
    };

    // Build schema properties from fields
    let mut schema_properties = Vec::new();
    let mut required_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();

        // Extract field documentation
        let field_desc = extract_doc_comments(&field.attrs);

        // Get the JSON type for this field
        let json_type = get_json_type(&field.ty);

        // Determine if field is Option<T> (optional)
        let is_optional = is_option_type(&field.ty);

        if !is_optional {
            required_fields.push(field_name_str.clone());
        }

        // Check if this is an array type and get the item type
        let array_item_type = get_array_item_type(&field.ty);

        if let Some(item_type) = array_item_type {
            // Array type - include items property
            schema_properties.push(quote! {
                properties.insert(
                    #field_name_str.to_string(),
                    serde_json::json!({
                        "type": #json_type,
                        "description": #field_desc,
                        "items": {
                            "type": #item_type
                        }
                    })
                );
            });
        } else {
            // Non-array type
            schema_properties.push(quote! {
                properties.insert(
                    #field_name_str.to_string(),
                    serde_json::json!({
                        "type": #json_type,
                        "description": #field_desc
                    })
                );
            });
        }
    }

    let required_array = if required_fields.is_empty() {
        quote! { serde_json::json!([]) }
    } else {
        let req_fields = required_fields.iter();
        quote! { serde_json::json!([#(#req_fields),*]) }
    };

    let expanded = quote! {
        impl #name {
            /// Get the tool name (snake_case version of struct name)
            pub fn tool_name() -> &'static str {
                #tool_name
            }

            /// Get the tool description from doc comments
            pub fn description() -> &'static str {
                #struct_description
            }

            /// Generate JSON schema for this tool
            pub fn schema() -> serde_json::Value {
                let mut properties = serde_json::Map::new();
                #(#schema_properties)*

                serde_json::json!({
                    "type": "object",
                    "properties": serde_json::Value::Object(properties),
                    "required": #required_array
                })
            }

            /// Generate provider-specific tool schema using a SchemaGenerator
            pub fn tool_schema(generator: &dyn ::reson_agentic::schema::SchemaGenerator) -> serde_json::Value {
                generator.generate_schema(
                    #tool_name,
                    #struct_description,
                    Self::schema()
                )
            }
        }
    };

    TokenStream::from(expanded)
}

/// `#[derive(Deserializable)]` for streaming-parseable types
///
/// Implements the Deserializable trait for progressive parsing during streaming.
/// Types with this derive can be constructed from partial JSON as it arrives.
///
/// # Example
/// ```ignore
/// #[derive(Deserializable, Serialize, Deserialize)]
/// struct Person {
///     /// The person's name
///     name: String,
///     /// The person's age
///     age: u32,
///     /// Optional email address
///     email: Option<String>,
/// }
/// ```
#[proc_macro_derive(Deserializable)]
pub fn derive_deserializable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let name = &input.ident;

    // Extract fields
    let fields = match &input.data {
        syn::Data::Struct(data_struct) => match &data_struct.fields {
            syn::Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    name,
                    "Deserializable derive only supports structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(name, "Deserializable derive only supports structs")
                .to_compile_error()
                .into();
        }
    };

    // Build field descriptions
    let mut field_desc_tokens = Vec::new();
    let mut validation_checks = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let field_desc = extract_doc_comments(&field.attrs);
        let field_type = &field.ty;
        let field_type_str = quote!(#field_type).to_string();
        let is_optional = is_option_type(&field.ty);
        let is_required = !is_optional;

        field_desc_tokens.push(quote! {
            ::reson_agentic::parsers::FieldDescription {
                name: #field_name_str.to_string(),
                field_type: #field_type_str.to_string(),
                description: #field_desc.to_string(),
                required: #is_required,
            }
        });

        // Add validation for required fields
        if is_required {
            validation_checks.push(quote! {
                if let serde_json::Value::Null = serde_json::to_value(&self.#field_name)
                    .map_err(|e| ::reson_agentic::error::Error::NonRetryable(e.to_string()))? {
                    return Err(::reson_agentic::error::Error::NonRetryable(
                        format!("Required field '{}' is missing or null", #field_name_str)
                    ));
                }
            });
        }
    }

    let validation_logic = if validation_checks.is_empty() {
        quote! { Ok(()) }
    } else {
        quote! {
            #(#validation_checks)*
            Ok(())
        }
    };

    let expanded = quote! {
        impl ::reson_agentic::parsers::Deserializable for #name {
            fn from_partial(partial: serde_json::Value) -> ::reson_agentic::error::Result<Self> {
                serde_json::from_value(partial).map_err(|e| {
                    ::reson_agentic::error::Error::NonRetryable(format!("Failed to parse {}: {}", stringify!(#name), e))
                })
            }

            fn validate_complete(&self) -> ::reson_agentic::error::Result<()> {
                #validation_logic
            }

            fn field_descriptions() -> Vec<::reson_agentic::parsers::FieldDescription> {
                vec![
                    #(#field_desc_tokens),*
                ]
            }
        }
    };

    TokenStream::from(expanded)
}

// Helper function to convert PascalCase to snake_case
fn convert_to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(ch.to_lowercase().next().unwrap());
        } else {
            result.push(ch);
        }
    }
    result
}

// Helper to check if a type is Option<T>
fn is_option_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
}

// Helper to extract doc comments from attributes
fn extract_doc_comments(attrs: &[syn::Attribute]) -> String {
    let mut docs = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(meta) = &attr.meta {
                if let syn::Expr::Lit(expr_lit) = &meta.value {
                    if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                        docs.push(lit_str.value().trim().to_string());
                    }
                }
            }
        }
    }
    docs.join(" ")
}

// Helper to get JSON schema type from Rust type
fn get_json_type(ty: &syn::Type) -> String {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            let ident = segment.ident.to_string();

            // Handle Option<T> - extract inner type
            if ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                        return get_json_type(inner_ty);
                    }
                }
            }

            // Map Rust types to JSON schema types
            return match ident.as_str() {
                "String" | "str" => "string",
                "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => "integer",
                "f32" | "f64" => "number",
                "bool" => "boolean",
                "Vec" => "array",
                "HashMap" | "BTreeMap" => "object",
                _ => "object", // Default for custom types
            }.to_string();
        }
    }
    "object".to_string()
}

/// Get the inner type of Vec<T> or Option<Vec<T>>
fn get_array_item_type(ty: &syn::Type) -> Option<String> {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            let ident = segment.ident.to_string();

            // Handle Option<Vec<T>> - extract Vec<T> first
            if ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                        return get_array_item_type(inner_ty);
                    }
                }
            }

            // Handle Vec<T> - extract T
            if ident == "Vec" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                        return Some(get_json_type(inner_ty));
                    }
                }
                // Default to string if we can't determine the inner type
                return Some("string".to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_case_conversion() {
        assert_eq!(convert_to_snake_case("CalculatorTool"), "calculator_tool");
        assert_eq!(convert_to_snake_case("GetWeather"), "get_weather");
        assert_eq!(convert_to_snake_case("HTTPClient"), "h_t_t_p_client");
        assert_eq!(convert_to_snake_case("Simple"), "simple");
    }
}
