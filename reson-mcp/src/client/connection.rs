//! MCP Client connection management

#[cfg(feature = "apps")]
use rmcp::model::ExtensionCapabilities;
use rmcp::{
    model::{
        CallToolRequestParams, CallToolResult, ClientInfo, Implementation, ListResourcesResult,
        ListToolsResult, ReadResourceRequestParams, ReadResourceResult, ServerInfo,
    },
    service::RunningService,
    transport::{ConfigureCommandExt, StreamableHttpClientTransport, TokioChildProcess},
    ClientHandler, RoleClient, ServiceExt,
};
use serde_json::Value;
use tokio::process::Command;

use crate::error::{Error, Result};

/// MCP client handler that advertises extension capabilities during initialize.
///
/// When the `apps` feature is enabled, advertises support for the
/// `io.modelcontextprotocol/ui` extension (SEP-1865) so servers can
/// expose UI-enabled tools.
struct ResonClientHandler;

impl ClientHandler for ResonClientHandler {
    fn get_info(&self) -> ClientInfo {
        #[cfg(feature = "apps")]
        let capabilities = {
            let mut ext = ExtensionCapabilities::new();
            let mut ui = serde_json::Map::new();
            ui.insert(
                "mimeTypes".to_string(),
                serde_json::json!(["text/html;profile=mcp-app"]),
            );
            ext.insert("io.modelcontextprotocol/ui".to_string(), ui);

            rmcp::model::ClientCapabilities::builder()
                .enable_extensions_with(ext)
                .build()
        };

        #[cfg(not(feature = "apps"))]
        let capabilities = rmcp::model::ClientCapabilities::default();

        ClientInfo {
            protocol_version: Default::default(),
            capabilities,
            client_info: Implementation {
                name: "reson-mcp".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Transport type for MCP connections
#[derive(Debug, Clone)]
pub enum Transport {
    /// HTTP/SSE streaming transport
    Http(String),
    /// WebSocket transport
    WebSocket(String),
    /// Stdio transport (spawns child process)
    Stdio { command: String, args: Vec<String> },
}

/// MCP Server configuration for registering with a runtime
#[derive(Debug, Clone)]
pub struct McpServer {
    transport: Transport,
}

impl McpServer {
    /// Create an MCP server configuration for HTTP transport
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpServer;
    ///
    /// let server = McpServer::http("http://localhost:8080/mcp");
    /// ```
    pub fn http(url: impl Into<String>) -> Self {
        Self {
            transport: Transport::Http(url.into()),
        }
    }

    /// Create an MCP server configuration for WebSocket transport
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpServer;
    ///
    /// let server = McpServer::websocket("ws://localhost:8080/mcp");
    /// ```
    pub fn websocket(url: impl Into<String>) -> Self {
        Self {
            transport: Transport::WebSocket(url.into()),
        }
    }

    /// Create an MCP server configuration for stdio transport
    ///
    /// The command string is parsed to extract the command and arguments.
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpServer;
    ///
    /// let server = McpServer::stdio("npx @modelcontextprotocol/server-filesystem /tmp");
    /// ```
    pub fn stdio(command_line: impl Into<String>) -> Self {
        let command_line = command_line.into();
        let parts: Vec<&str> = command_line.split_whitespace().collect();
        let (command, args) = if parts.is_empty() {
            (command_line.clone(), vec![])
        } else {
            (
                parts[0].to_string(),
                parts[1..].iter().map(|s| s.to_string()).collect(),
            )
        };

        Self {
            transport: Transport::Stdio { command, args },
        }
    }

    /// Create an MCP server configuration for stdio with explicit command and args
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpServer;
    ///
    /// let server = McpServer::stdio_with_args("npx", &["@modelcontextprotocol/server-filesystem", "/tmp"]);
    /// ```
    pub fn stdio_with_args(command: impl Into<String>, args: &[impl AsRef<str>]) -> Self {
        Self {
            transport: Transport::Stdio {
                command: command.into(),
                args: args.iter().map(|s| s.as_ref().to_string()).collect(),
            },
        }
    }

    /// Get the transport configuration
    pub fn transport(&self) -> &Transport {
        &self.transport
    }

    /// Connect to the MCP server and return a client
    pub async fn connect(self) -> Result<McpClient> {
        McpClient::from_server(self).await
    }
}

/// An active MCP client connection
pub struct McpClient {
    inner: ClientInner,
}

enum ClientInner {
    Http(RunningService<RoleClient, ResonClientHandler>),
    WebSocket(RunningService<RoleClient, ResonClientHandler>),
    Stdio(RunningService<RoleClient, ResonClientHandler>),
}

impl McpClient {
    /// Connect to an MCP server via HTTP
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpClient;
    ///
    /// # async fn example() -> reson_mcp::Result<()> {
    /// let client = McpClient::http("http://localhost:8080/mcp").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn http(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        let transport = StreamableHttpClientTransport::from_uri(url.clone());
        let service = ResonClientHandler.serve(transport).await.map_err(|e| {
            Error::Transport(format!(
                "Failed to connect to HTTP MCP server at {}: {}",
                url, e
            ))
        })?;
        Ok(Self {
            inner: ClientInner::Http(service),
        })
    }

    /// Connect to an MCP server via WebSocket
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpClient;
    ///
    /// # async fn example() -> reson_mcp::Result<()> {
    /// let client = McpClient::websocket("ws://localhost:8080/mcp").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn websocket(url: impl Into<String>) -> Result<Self> {
        let url = url.into();
        let transport = crate::transport::websocket::connect(&url).await?;
        let service = ResonClientHandler.serve(transport).await.map_err(|e| {
            Error::Transport(format!(
                "Failed to connect to WebSocket MCP server at {}: {}",
                url, e
            ))
        })?;
        Ok(Self {
            inner: ClientInner::WebSocket(service),
        })
    }

    /// Connect to an MCP server via stdio (spawns child process)
    ///
    /// The command string is parsed to extract the command and arguments.
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpClient;
    ///
    /// # async fn example() -> reson_mcp::Result<()> {
    /// let client = McpClient::stdio("npx @modelcontextprotocol/server-filesystem /tmp").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stdio(command_line: impl Into<String>) -> Result<Self> {
        let command_line = command_line.into();
        let parts: Vec<&str> = command_line.split_whitespace().collect();
        if parts.is_empty() {
            return Err(Error::Transport("Empty command".to_string()));
        }

        let command = parts[0];
        let args: Vec<&str> = parts[1..].to_vec();

        let transport = TokioChildProcess::new(Command::new(command).configure(|cmd| {
            cmd.args(&args);
        }))
        .map_err(|e| Error::Transport(format!("Failed to spawn process '{}': {}", command, e)))?;

        let service = ResonClientHandler.serve(transport).await.map_err(|e| {
            Error::Transport(format!(
                "Failed to connect to stdio MCP server '{}': {}",
                command_line, e
            ))
        })?;

        Ok(Self {
            inner: ClientInner::Stdio(service),
        })
    }

    /// Create a client from an McpServer configuration
    async fn from_server(server: McpServer) -> Result<Self> {
        match server.transport {
            Transport::Http(url) => Self::http(url).await,
            Transport::WebSocket(url) => Self::websocket(url).await,
            Transport::Stdio { command, args } => Self::stdio_with_args(&command, &args).await,
        }
    }

    /// Connect to an MCP server via stdio with explicit command and args
    pub async fn stdio_with_args(command: &str, args: &[String]) -> Result<Self> {
        let transport = TokioChildProcess::new(Command::new(command).configure(|cmd| {
            cmd.args(args);
        }))
        .map_err(|e| Error::Transport(format!("Failed to spawn process '{}': {}", command, e)))?;

        let service = ResonClientHandler.serve(transport).await.map_err(|e| {
            Error::Transport(format!(
                "Failed to connect to stdio MCP server '{}': {}",
                command, e
            ))
        })?;

        Ok(Self {
            inner: ClientInner::Stdio(service),
        })
    }

    /// Get information about the connected server
    ///
    /// Returns `None` if the server info is not yet available (shouldn't happen
    /// after successful connection).
    pub fn server_info(&self) -> Option<&ServerInfo> {
        self.service().peer_info()
    }

    /// List all available tools from the MCP server
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpClient;
    ///
    /// # async fn example() -> reson_mcp::Result<()> {
    /// let client = McpClient::http("http://localhost:8080/mcp").await?;
    /// let tools = client.list_tools().await?;
    /// for tool in &tools.tools {
    ///     println!("Tool: {} - {:?}", tool.name, tool.description);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_tools(&self) -> Result<ListToolsResult> {
        self.service()
            .list_tools(Default::default())
            .await
            .map_err(|e| Error::Protocol(format!("Failed to list tools: {}", e)))
    }

    /// Call a tool on the MCP server
    ///
    /// # Arguments
    /// * `name` - The name of the tool to call
    /// * `arguments` - The arguments to pass to the tool as a JSON value
    ///
    /// # Example
    /// ```rust,no_run
    /// use reson_mcp::client::McpClient;
    /// use serde_json::json;
    ///
    /// # async fn example() -> reson_mcp::Result<()> {
    /// let client = McpClient::http("http://localhost:8080/mcp").await?;
    /// let result = client.call_tool("read_file", json!({"path": "/tmp/test.txt"})).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn call_tool(
        &self,
        name: impl Into<String>,
        arguments: Value,
    ) -> Result<CallToolResult> {
        let name = name.into();
        let arguments = arguments.as_object().cloned();

        self.service()
            .call_tool(CallToolRequestParams {
                meta: None,
                name: name.clone().into(),
                arguments,
                task: None,
            })
            .await
            .map_err(|e| Error::ToolExecution(format!("Failed to call tool '{}': {}", name, e)))
    }

    /// List all available resources from the MCP server
    pub async fn list_resources(&self) -> Result<ListResourcesResult> {
        self.service()
            .list_resources(Default::default())
            .await
            .map_err(|e| Error::Protocol(format!("Failed to list resources: {}", e)))
    }

    /// Read a resource by URI from the MCP server
    pub async fn read_resource(&self, uri: impl Into<String>) -> Result<ReadResourceResult> {
        let uri = uri.into();
        self.service()
            .read_resource(ReadResourceRequestParams {
                meta: None,
                uri: uri.clone(),
            })
            .await
            .map_err(|e| Error::Protocol(format!("Failed to read resource '{}': {}", uri, e)))
    }

    /// Gracefully close the connection
    pub async fn close(self) -> Result<()> {
        match self.inner {
            ClientInner::Http(service)
            | ClientInner::WebSocket(service)
            | ClientInner::Stdio(service) => {
                service
                    .cancel()
                    .await
                    .map_err(|e| Error::Transport(format!("Failed to close connection: {}", e)))?;
                Ok(())
            }
        }
    }

    fn service(&self) -> &RunningService<RoleClient, ResonClientHandler> {
        match &self.inner {
            ClientInner::Http(s) | ClientInner::WebSocket(s) | ClientInner::Stdio(s) => s,
        }
    }
}
