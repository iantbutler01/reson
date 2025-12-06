//! Services module - inference_clients and related types

use pyo3::prelude::*;
use pyo3::types::PyList;
use reson::utils::ConversationMessage;

use crate::types::{ChatMessage, ToolCall, ToolResult, ReasoningSegment};

/// Inference provider enum matching the Python InferenceProvider
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InferenceProvider {
    OPENAI,
    ANTHROPIC,
    GOOGLE_GENAI,
    OPENROUTER,
    BEDROCK,
    GOOGLE_ANTHROPIC,
}

#[pymethods]
impl InferenceProvider {
    fn __repr__(&self) -> String {
        match self {
            InferenceProvider::OPENAI => "InferenceProvider.OPENAI".to_string(),
            InferenceProvider::ANTHROPIC => "InferenceProvider.ANTHROPIC".to_string(),
            InferenceProvider::GOOGLE_GENAI => "InferenceProvider.GOOGLE_GENAI".to_string(),
            InferenceProvider::OPENROUTER => "InferenceProvider.OPENROUTER".to_string(),
            InferenceProvider::BEDROCK => "InferenceProvider.BEDROCK".to_string(),
            InferenceProvider::GOOGLE_ANTHROPIC => "InferenceProvider.GOOGLE_ANTHROPIC".to_string(),
        }
    }
}

impl From<InferenceProvider> for reson::types::Provider {
    fn from(provider: InferenceProvider) -> Self {
        match provider {
            InferenceProvider::OPENAI => reson::types::Provider::OpenAI,
            InferenceProvider::ANTHROPIC => reson::types::Provider::Anthropic,
            InferenceProvider::GOOGLE_GENAI => reson::types::Provider::GoogleGenAI,
            InferenceProvider::OPENROUTER => reson::types::Provider::OpenRouter,
            InferenceProvider::BEDROCK => reson::types::Provider::Bedrock,
            InferenceProvider::GOOGLE_ANTHROPIC => reson::types::Provider::GoogleAnthropic,
        }
    }
}

impl From<reson::types::Provider> for InferenceProvider {
    fn from(provider: reson::types::Provider) -> Self {
        match provider {
            reson::types::Provider::OpenAI => InferenceProvider::OPENAI,
            reson::types::Provider::Anthropic => InferenceProvider::ANTHROPIC,
            reson::types::Provider::GoogleGenAI => InferenceProvider::GOOGLE_GENAI,
            reson::types::Provider::OpenRouter => InferenceProvider::OPENROUTER,
            reson::types::Provider::Bedrock => InferenceProvider::BEDROCK,
            reson::types::Provider::GoogleAnthropic => InferenceProvider::GOOGLE_ANTHROPIC,
        }
    }
}

/// Base class for inference clients - provides message conversion functionality
#[pyclass(subclass)]
pub struct InferenceClient {
    // Base class has no fields - subclasses add provider-specific config
}

#[pymethods]
impl InferenceClient {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, pyo3::types::PyTuple>, _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> Self {
        Self {}
    }

    /// Convert mixed ChatMessage/ToolResult/ToolCall/ReasoningSegment list to provider-specific format.
    ///
    /// Provider-aware behavior:
    /// - Anthropic/Bedrock/GoogleAnthropic: Coalesce consecutive ToolResults into a single user turn
    ///   with a content array of tool_result blocks.
    /// - Google GenAI: Coalesce consecutive ToolResults into a single user turn with functionResponse parts.
    /// - OpenAI/OpenRouter: Map each ToolResult to a separate tool role message.
    #[pyo3(signature = (messages, provider))]
    fn _convert_messages_to_provider_format(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyList>,
        provider: InferenceProvider,
    ) -> PyResult<PyObject> {
        // Convert Python messages to Rust ConversationMessage types
        let mut conversation_messages: Vec<ConversationMessage> = Vec::new();

        for item in messages.iter() {
            // Try to extract as each type
            if let Ok(chat_msg) = item.extract::<ChatMessage>() {
                let rust_msg: reson::types::ChatMessage = chat_msg.into();
                conversation_messages.push(ConversationMessage::Chat(rust_msg));
            } else if let Ok(tool_call) = item.extract::<ToolCall>() {
                let rust_tc: reson::types::ToolCall = tool_call.into();
                conversation_messages.push(ConversationMessage::ToolCall(rust_tc));
            } else if let Ok(tool_result) = item.extract::<ToolResult>() {
                let rust_tr: reson::types::ToolResult = tool_result.into();
                conversation_messages.push(ConversationMessage::ToolResult(rust_tr));
            } else if let Ok(reasoning) = item.extract::<ReasoningSegment>() {
                let rust_rs: reson::types::ReasoningSegment = reasoning.into();
                conversation_messages.push(ConversationMessage::Reasoning(rust_rs));
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    format!("Unsupported message type: {:?}", item.get_type().name())
                ));
            }
        }

        // Convert using the Rust function
        let rust_provider: reson::types::Provider = provider.into();
        let converted = reson::utils::convert_messages_to_provider_format(
            &conversation_messages,
            rust_provider,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Convert Vec<Value> back to Python list of dicts
        pythonize::pythonize(py, &converted)
            .map(|bound| bound.unbind())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}
