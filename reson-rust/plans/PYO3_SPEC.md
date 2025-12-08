# PyO3 Bindings Specification for reson_agentic

## Overview

This document specifies the Python bindings needed to replace the pure-Python reson implementation with Rust via PyO3. The goal is to run existing Python integration tests unchanged.

## Architecture

```
reson_agentic/
├── src/                    # Core Rust library
├── reson-macros/           # Proc macros (#[agentic], #[tool], etc.)
└── reson-py/               # NEW: PyO3 bindings crate
    ├── Cargo.toml
    └── src/
        ├── lib.rs          # Module entry point
        ├── runtime.rs      # Runtime class
        ├── types.rs        # ChatMessage, ToolCall, ToolResult, etc.
        ├── stores.rs       # Store implementations
        ├── decorators.rs   # @agentic, @agentic_generator
        └── errors.rs       # Exception classes
```

## Python Module Structure

```python
# After: import reson
from reson import (
    # Decorators
    agentic,
    agentic_generator,

    # Core class
    Runtime,

    # Types (re-exported from reson.types)
    ChatMessage,
    ChatRole,
    ToolCall,
    ToolResult,
    ReasoningSegment,
    Deserializable,

    # Stores
    MemoryStore,
    RedisStore,
    PostgresStore,
    MemoryStoreConfig,
    RedisStoreConfig,
    PostgresStoreConfig,
)
```

---

## 1. Core Classes

### 1.1 Runtime

```python
#[pyclass]
class Runtime:
    """Main runtime for agentic functions."""

    # Constructor
    def __new__(
        cls,
        model: Optional[str] = None,
        store: Optional[Store] = None,
        used: bool = False,
        api_key: Optional[str] = None,
        native_tools: bool = False,
    ) -> Runtime: ...

    # Properties (read-only)
    @property
    def raw_response(self) -> str: ...

    @property
    def reasoning(self) -> str: ...

    @property
    def reasoning_segments(self) -> List[ReasoningSegment]: ...

    @property
    def native_tools(self) -> bool: ...

    @property
    def model(self) -> Optional[str]: ...

    # Methods
    def clear_raw_response(self) -> None: ...
    def clear_reasoning(self) -> None: ...
    def clear_reasoning_segments(self) -> None: ...

    # Tool registration
    def tool(
        self,
        fn: Callable,
        *,
        name: Optional[str] = None,
        tool_type: Optional[Type[Deserializable]] = None,
    ) -> None: ...

    # Tool detection & execution
    def is_tool_call(self, result: Any) -> bool: ...
    def get_tool_name(self, result: Any) -> Optional[str]: ...
    async def execute_tool(self, tool_result: Any) -> Any: ...

    # Main inference (ASYNC)
    async def run(
        self,
        *,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[List[Union[ChatMessage, ToolResult, ReasoningSegment]]] = None,
        output_type: Optional[Type] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Any: ...

    # Streaming inference (ASYNC GENERATOR)
    async def run_stream(
        self,
        *,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[List[Union[ChatMessage, ToolResult, ReasoningSegment]]] = None,
        output_type: Optional[Type] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AsyncIterator[Tuple[str, Any]]: ...
    # Yields: ("content", str), ("reasoning", str), ("tool_call_complete", ToolCall), etc.
```

### 1.2 Implementation Notes for Runtime

- **Async Methods**: Use `pyo3-asyncio` with tokio runtime
- **Tool Registration**: Store Python callables, invoke via `py.call()`
- **Streaming**: Return `PyAsyncIterator` that yields tuples
- **Internal State**:
  - `_tools: HashMap<String, PyObject>`
  - `_raw_response_accumulator: Vec<String>`
  - `_reasoning_accumulator: Vec<String>`
  - `_reasoning_segments: Vec<ReasoningSegment>`

---

## 2. Data Types

### 2.1 ChatRole (Enum)

```python
#[pyclass]
class ChatRole:
    SYSTEM: ClassVar[ChatRole]
    USER: ClassVar[ChatRole]
    ASSISTANT: ClassVar[ChatRole]
    TOOL_RESULT: ClassVar[ChatRole]

    @property
    def value(self) -> str: ...
```

### 2.2 ChatMessage

```python
#[pyclass]
class ChatMessage:
    role: ChatRole
    content: str
    cache_marker: bool
    signature: Optional[str]

    def __new__(
        cls,
        role: ChatRole,
        content: str,
        cache_marker: bool = False,
        signature: Optional[str] = None,
    ) -> ChatMessage: ...

    # Convenience constructors
    @classmethod
    def user(cls, content: str) -> ChatMessage: ...

    @classmethod
    def assistant(cls, content: str) -> ChatMessage: ...

    @classmethod
    def system(cls, content: str) -> ChatMessage: ...

    # Serialization
    def model_dump(self) -> Dict[str, Any]: ...

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> ChatMessage: ...
```

### 2.3 ToolCall

```python
#[pyclass]
class ToolCall:
    tool_use_id: str
    tool_name: str
    args: Optional[Dict[str, Any]]
    raw_arguments: Optional[str]
    signature: Optional[str]

    def __new__(
        cls,
        tool_use_id: str,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        raw_arguments: Optional[str] = None,
        signature: Optional[str] = None,
    ) -> ToolCall: ...

    @classmethod
    def create(
        cls,
        tool_call_obj: Any,
        signature: Optional[str] = None,
    ) -> Union[ToolCall, List[ToolCall]]: ...

    def to_provider_format(self, provider: str) -> Dict[str, Any]: ...
    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCall: ...
```

### 2.4 ToolResult

```python
#[pyclass]
class ToolResult:
    tool_use_id: str
    content: str
    is_error: bool
    signature: Optional[str]
    tool_name: Optional[str]

    def __new__(
        cls,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
        signature: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> ToolResult: ...

    @classmethod
    def create(
        cls,
        tool_call_obj: Any,
        result: str,
        signature: Optional[str] = None,
    ) -> Union[ToolResult, List[ToolResult]]: ...

    def to_provider_format(self, provider: str) -> Dict[str, Any]: ...
    def to_chat_message(self) -> ChatMessage: ...
    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolResult: ...
```

### 2.5 ReasoningSegment

```python
#[pyclass]
class ReasoningSegment:
    content: str
    signature: Optional[str]
    provider_metadata: Dict[str, Any]
    segment_index: int

    def __new__(
        cls,
        content: str,
        signature: Optional[str] = None,
        provider_metadata: Optional[Dict[str, Any]] = None,
        segment_index: int = 0,
    ) -> ReasoningSegment: ...

    def to_provider_format(self, provider: str) -> Dict[str, Any]: ...
```

### 2.6 Deserializable (Abstract Base)

```python
# This is tricky - Python users define Deserializable subclasses
# We need to support arbitrary Python classes that have certain methods

class Deserializable(Protocol):
    """Protocol for types that can be deserialized from LLM output."""

    @classmethod
    def from_partial(cls, value: Dict[str, Any]) -> Self: ...

    def validate_complete(self) -> None: ...

    @classmethod
    def field_descriptions(cls) -> List[FieldDescription]: ...
```

**Implementation**: Use `gasp` Python package or port the deserializable logic. For PyO3, we accept any Python class and check for these methods at runtime.

---

## 3. Store Classes

### 3.1 Store Protocol

```python
class Store(Protocol):
    async def get(self, key: str, default: Any = None) -> Any: ...
    async def set(self, key: str, value: Any) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def get_all(self) -> Dict[str, Any]: ...
    async def keys(self) -> Set[str]: ...
    async def close(self) -> None: ...
    async def publish_to_mailbox(self, mailbox_id: str, message: Any) -> None: ...
    async def get_message(self, mailbox_id: str, timeout: Optional[float] = None) -> Optional[Any]: ...
```

### 3.2 MemoryStore

```python
#[pyclass]
class MemoryStore:
    def __new__(cls) -> MemoryStore: ...

    # Implements Store protocol
    async def get(self, key: str, default: Any = None) -> Any: ...
    async def set(self, key: str, value: Any) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
    async def get_all(self) -> Dict[str, Any]: ...
    async def keys(self) -> Set[str]: ...
    async def close(self) -> None: ...
    async def publish_to_mailbox(self, mailbox_id: str, message: Any) -> None: ...
    async def get_message(self, mailbox_id: str, timeout: Optional[float] = None) -> Optional[Any]: ...
```

### 3.3 RedisStore

```python
#[pyclass]
class RedisStore:
    def __new__(
        cls,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ) -> RedisStore: ...

    # Implements Store protocol
    ...
```

### 3.4 PostgresStore

```python
#[pyclass]
class PostgresStore:
    def __new__(
        cls,
        dsn: str,
        table: str,
        column: str,
    ) -> PostgresStore: ...

    # Implements Store protocol
    ...
```

### 3.5 Store Configs (for decorator)

```python
#[pyclass]
class MemoryStoreConfig:
    kind: str = "memory"

#[pyclass]
class RedisStoreConfig:
    kind: str = "redis"
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

#[pyclass]
class PostgresStoreConfig:
    kind: str = "postgres"
    dsn: str
    table: str
    column: str
```

---

## 4. Decorators

### 4.1 @agentic

```python
def agentic(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = False,  # Deprecated, ignored
    native_tools: bool = False,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for async agentic functions.

    Injects a Runtime instance as the 'runtime' parameter.
    Function docstring becomes the default prompt.
    """
    ...
```

**Implementation**:
1. Accept async Python function
2. Create wrapper that:
   - Creates Runtime with specified config
   - Injects runtime as `runtime` parameter
   - Sets `runtime._default_prompt` from docstring
   - Calls original function
   - Validates `runtime.used` is True
   - Returns result

### 4.2 @agentic_generator

```python
def agentic_generator(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = False,  # Deprecated, ignored
    native_tools: bool = False,
) -> Callable[[Callable[..., AsyncGenerator[T, None]]], Callable[..., AsyncGenerator[T, None]]]:
    """
    Decorator for async generator agentic functions.
    """
    ...
```

---

## 5. Exception Classes

```python
#[pyclass(extends=PyException)]
class NonRetryableException(Exception):
    """Error that should not be retried."""
    pass

#[pyclass(extends=PyException)]
class InferenceException(Exception):
    """Generic inference error, safe to retry."""
    pass

#[pyclass(extends=InferenceException)]
class ContextLengthExceeded(InferenceException, ValueError):
    """Context length exceeded."""
    pass

#[pyclass(extends=InferenceException)]
class RetriesExceeded(InferenceException):
    """Maximum retries exceeded."""
    pass
```

---

## 6. Key Implementation Details

### 6.1 Async Runtime

Use `pyo3-asyncio-0.21` with tokio:

```rust
use pyo3::prelude::*;
use pyo3_asyncio::tokio::future_into_py;

#[pymethods]
impl Runtime {
    fn run<'py>(&self, py: Python<'py>, /* args */) -> PyResult<&'py PyAny> {
        let runtime = self.inner.clone();
        future_into_py(py, async move {
            let result = runtime.run(/* args */).await?;
            Ok(result.into_py(py))
        })
    }
}
```

### 6.2 Async Generators (Streaming)

For `run_stream`, return a Python async iterator:

```rust
#[pyclass]
struct StreamIterator {
    receiver: tokio::sync::mpsc::Receiver<StreamChunk>,
}

#[pymethods]
impl StreamIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<&'py PyAny>> {
        // Return next chunk or None when done
    }
}
```

### 6.3 Calling Python Callables (Tools)

```rust
fn execute_tool(&self, py: Python<'_>, tool_name: &str, args: PyObject) -> PyResult<PyObject> {
    let tool_fn = self.tools.get(tool_name)?;
    tool_fn.call(py, (args,), None)
}
```

### 6.4 Pydantic Compatibility

All classes should support:
- `model_dump()` -> `Dict[str, Any]`
- `model_validate(data: Dict)` -> `Self`
- Proper `__repr__` and `__eq__`

---

## 7. Cargo.toml for reson-py

```toml
[package]
name = "reson-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "reson"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
pyo3-asyncio = { version = "0.21", features = ["tokio-runtime"] }
tokio = { version = "1.40", features = ["full"] }
reson = { path = ".." }  # Core Rust library

[build]
# Build with maturin
```

---

## 8. Testing Strategy

1. **Unit Tests**: Rust tests for PyO3 bindings
2. **Integration Tests**: Run existing Python tests unchanged:
   ```bash
   cd /Users/crow/SoftwareProjects/reson
   pip install ./reson_agentic/reson-py  # Install Rust bindings
   pytest integration_tests/          # Run existing tests
   ```

---

## 9. Migration Path

1. Build `reson-py` with maturin
2. Install as `reson` package (replaces pure Python)
3. Run integration tests
4. Fix any compatibility issues
5. Remove pure Python `reson/` directory

---

## 10. Not Included (Separate Concerns)

- **DatabaseManager / ORM** (`reson/data/postgres/`) - Separate package
- **TrainingManager** - Can be added later if needed
- **BAML integration** - Optional, can be added later
- **ART backend** - Optional, specific to certain use cases
