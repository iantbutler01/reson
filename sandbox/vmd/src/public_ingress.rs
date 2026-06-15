use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::{
    Router,
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::any,
};
use hyper::Request as HyperRequest;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::{RwLock, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::config::Config;
use crate::state::manager::Manager;
use crate::state::types::{VmMetadata, VmState};

const VM_METADATA_PUBLIC_INGRESSES: &str = "chevalier.public_ingresses";
const OWNER_BEARER_HEADER: &str = "x-chevalier-owner-bearer";
const OWNER_COOKIE_HEADER: &str = "x-chevalier-owner-cookie";
const INGRESS_TOKEN_HEADER: &str = "x-chevalier-ingress-token";
const INGRESS_TOKEN_QUERY_PARAM: &str = "chevalier_ingress_token";
const ROUTE_CACHE_TTL: Duration = Duration::from_secs(1);
const DENIED_PUBLIC_GUEST_PORTS: &[u16] =
    &[22, 11211, 2375, 2376, 3306, 5432, 6379, 13337, 13338, 27017];

pub struct PublicIngressHandle {
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl PublicIngressHandle {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

#[derive(Clone)]
struct PublicIngressState {
    manager: Arc<Manager>,
    cfg: Config,
    http_client: reqwest::Client,
    route_cache: Arc<RwLock<RouteCache>>,
}

#[derive(Debug, Clone, Deserialize)]
struct PublicIngressExposure {
    owner_id: Uuid,
    public_hostname: String,
    guest_port: u16,
    #[serde(default = "default_auth_required")]
    auth_required: bool,
}

#[derive(Debug, Clone)]
struct ResolvedIngressRoute {
    owner_id: Uuid,
    guest_port: u16,
    auth_required: bool,
    proxy_port: u16,
}

#[derive(Debug, Default)]
struct RouteCache {
    refreshed_at: Option<Instant>,
    routes: HashMap<String, ResolvedIngressRoute>,
}

pub async fn start(config: &Config, manager: Arc<Manager>) -> Result<Option<PublicIngressHandle>> {
    let Some(bind_addr) = config.network_services.public_ingress_bind_addr.as_deref() else {
        return Ok(None);
    };
    let bind_addr = bind_addr
        .parse::<SocketAddr>()
        .with_context(|| format!("parse public ingress bind addr {bind_addr}"))?;

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("bind public ingress listener {bind_addr}"))?;
    let state = PublicIngressState {
        manager,
        cfg: config.clone(),
        http_client: reqwest::Client::new(),
        route_cache: Arc::new(RwLock::new(RouteCache::default())),
    };
    let app = Router::new()
        .fallback(any(handle_public_ingress))
        .with_state(state);

    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        let server = axum::serve(listener, app).with_graceful_shutdown(async move {
            let _ = stop_rx.await;
        });
        if let Err(error) = server.await {
            warn!(error = %error, "public ingress server exited");
        }
    });

    Ok(Some(PublicIngressHandle {
        stop_tx: Some(stop_tx),
        join: Some(join),
    }))
}

async fn handle_public_ingress(State(state): State<PublicIngressState>, req: Request) -> Response {
    match handle_public_ingress_inner(state, req).await {
        Ok(response) => response,
        Err(error) => {
            let (status, body) = match &error {
                PublicIngressError::BadRequest(message) => {
                    (StatusCode::BAD_REQUEST, message.clone())
                }
                PublicIngressError::Unauthorized(message) => {
                    (StatusCode::UNAUTHORIZED, message.clone())
                }
                PublicIngressError::NotFound(message) => (StatusCode::NOT_FOUND, message.clone()),
                PublicIngressError::Upstream(error) => (
                    StatusCode::BAD_GATEWAY,
                    format!("public ingress proxy failed: {error}"),
                ),
            };
            warn!(status = %status, error = %error, "public ingress request failed");
            (status, body).into_response()
        }
    }
}

async fn handle_public_ingress_inner(
    state: PublicIngressState,
    req: Request,
) -> Result<Response, PublicIngressError> {
    let host_header = req
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let hostname = normalize_host(host_header);
    if hostname.is_empty() {
        return Err(PublicIngressError::BadRequest(
            "missing Host header".to_string(),
        ));
    }

    let Some(route) = resolve_route(&state, &hostname)
        .await
        .map_err(PublicIngressError::Upstream)?
    else {
        return Err(PublicIngressError::NotFound(
            "unknown ingress host".to_string(),
        ));
    };

    if route.auth_required {
        authorize_owner(
            &state,
            route.owner_id,
            extract_bearer_token(req.headers()).as_deref(),
            req.headers()
                .get(header::COOKIE)
                .and_then(|value| value.to_str().ok()),
            extract_ingress_token_from_query(req.uri()).as_deref(),
        )
        .await
        .map_err(PublicIngressError::Unauthorized)?;
    }

    proxy_request(route, req)
        .await
        .map_err(PublicIngressError::Upstream)
}

async fn resolve_route(
    state: &PublicIngressState,
    hostname: &str,
) -> Result<Option<ResolvedIngressRoute>> {
    {
        let cache = state.route_cache.read().await;
        if cache
            .refreshed_at
            .is_some_and(|refreshed_at| refreshed_at.elapsed() < ROUTE_CACHE_TTL)
        {
            return Ok(cache.routes.get(hostname).cloned());
        }
    }

    let mut cache = state.route_cache.write().await;
    if cache
        .refreshed_at
        .is_some_and(|refreshed_at| refreshed_at.elapsed() < ROUTE_CACHE_TTL)
    {
        return Ok(cache.routes.get(hostname).cloned());
    }

    let mut routes = HashMap::new();
    for vm in state.manager.list().await {
        if !vm_state_allows_public_ingress(vm.state) {
            continue;
        }
        let Ok(proxy_port) = u16::try_from(vm.network.proxy_port) else {
            continue;
        };
        if proxy_port == 0 {
            continue;
        }
        let Some(exposures) = decode_public_ingresses(&vm) else {
            continue;
        };
        for exposure in exposures {
            if !public_guest_port_allowed(exposure.guest_port) {
                warn!(
                    owner_id = %exposure.owner_id,
                    guest_port = exposure.guest_port,
                    hostname = %exposure.public_hostname,
                    "ignoring public ingress exposure for denied guest port"
                );
                continue;
            }
            routes.insert(
                normalize_host(&exposure.public_hostname),
                ResolvedIngressRoute {
                    owner_id: exposure.owner_id,
                    guest_port: exposure.guest_port,
                    auth_required: exposure.auth_required,
                    proxy_port,
                },
            );
        }
    }
    cache.refreshed_at = Some(Instant::now());
    cache.routes = routes;
    Ok(cache.routes.get(hostname).cloned())
}

fn vm_state_allows_public_ingress(state: VmState) -> bool {
    matches!(state, VmState::Running)
}

fn decode_public_ingresses(vm: &VmMetadata) -> Option<Vec<PublicIngressExposure>> {
    let raw = vm.metadata.get(VM_METADATA_PUBLIC_INGRESSES)?;
    serde_json::from_str::<Vec<PublicIngressExposure>>(raw).ok()
}

fn public_guest_port_allowed(guest_port: u16) -> bool {
    guest_port != 0 && !DENIED_PUBLIC_GUEST_PORTS.contains(&guest_port)
}

fn default_auth_required() -> bool {
    true
}

async fn authorize_owner(
    state: &PublicIngressState,
    owner_id: Uuid,
    bearer_token: Option<&str>,
    cookie_header: Option<&str>,
    ingress_token: Option<&str>,
) -> Result<(), String> {
    let user_token = bearer_token
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| extract_access_token_from_cookie(cookie_header));
    let ingress_token = ingress_token
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    if user_token.is_none() && ingress_token.is_none() {
        return Err("missing owner auth for protected ingress".to_string());
    }
    let service_token = state
        .cfg
        .vfs_internal_service_token
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "missing CHEVALIER_SANDBOX_VFS_INTERNAL_SERVICE_TOKEN".to_string())?;
    let base_url = state
        .cfg
        .network_services
        .api_internal_base_url
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "missing CHEVALIER_SANDBOX_API_INTERNAL_BASE_URL".to_string())?;
    let url = format!(
        "{}/v1/internal/vm-ingress/{}/authorize",
        base_url.trim_end_matches('/'),
        owner_id
    );
    let mut request = state
        .http_client
        .get(url)
        .header(header::AUTHORIZATION, format!("Bearer {service_token}"));
    if let Some(token) = user_token {
        request = request.header(OWNER_BEARER_HEADER, token);
    }
    if let Some(cookie) = cookie_header {
        request = request.header(OWNER_COOKIE_HEADER, cookie);
    }
    if let Some(token) = ingress_token {
        request = request.header(INGRESS_TOKEN_HEADER, token);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("authorize vm ingress owner: {error}"))?;
    if response.status().is_success() {
        return Ok(());
    }
    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Err("unauthorized ingress access".to_string());
    }
    Err(format!(
        "vm ingress auth failed with status {}",
        response.status()
    ))
}

async fn proxy_request(route: ResolvedIngressRoute, req: Request) -> Result<Response> {
    let (parts, body) = req.into_parts();

    let mut stream = TcpStream::connect(("127.0.0.1", route.proxy_port))
        .await
        .with_context(|| format!("connect local vm proxy port {}", route.proxy_port))?;
    stream
        .write_all(&route.guest_port.to_be_bytes())
        .await
        .with_context(|| format!("write guest port preface {}", route.guest_port))?;

    let io = TokioIo::new(stream);
    let (mut sender, connection) = http1::Builder::new()
        .handshake(io)
        .await
        .context("handshake http1 guest ingress upstream")?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            debug!(error = %error, "guest ingress upstream connection closed");
        }
    });

    let uri = sanitized_upstream_path_and_query(&parts.uri)
        .parse::<axum::http::uri::PathAndQuery>()
        .context("parse guest ingress upstream uri")?;

    let mut builder = HyperRequest::builder()
        .method(parts.method.clone())
        .uri(uri)
        .version(hyper::Version::HTTP_11);
    for (name, value) in &parts.headers {
        if should_skip_request_header(name) {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder = builder.header("x-forwarded-proto", "https");
    if let Some(host) = parts.headers.get(header::HOST) {
        builder = builder.header("x-forwarded-host", host);
    }

    let upstream_request = builder
        .body(body)
        .context("build guest ingress upstream request")?;
    let upstream_response = sender
        .send_request(upstream_request)
        .await
        .context("send guest ingress upstream request")?;

    let status = upstream_response.status();
    let headers = upstream_response.headers().clone();
    let response_body = Body::new(upstream_response.into_body());

    let mut response = Response::builder().status(status);
    if let Some(headers_mut) = response.headers_mut() {
        copy_response_headers(headers_mut, &headers);
    }
    response
        .body(Body::from(response_body))
        .context("build public ingress response")
}

fn copy_response_headers(target: &mut HeaderMap, source: &HeaderMap) {
    for (name, value) in source {
        if *name == header::CONTENT_LENGTH
            || *name == header::TRANSFER_ENCODING
            || *name == header::CONNECTION
        {
            continue;
        }
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            target.append(name, value);
        }
    }
}

fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(header::AUTHORIZATION)?.to_str().ok()?.trim();
    raw.strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
        .map(str::to_string)
}

fn extract_access_token_from_cookie(cookie_header: Option<&str>) -> Option<String> {
    cookie_header?
        .split(';')
        .filter_map(|cookie| cookie.trim().split_once('='))
        .find_map(|(name, value)| {
            let normalized = name.trim();
            if normalized.eq_ignore_ascii_case("access_token")
                || normalized.eq_ignore_ascii_case("accessToken")
            {
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            } else {
                None
            }
        })
}

fn extract_ingress_token_from_query(uri: &axum::http::Uri) -> Option<String> {
    uri.query()?.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        if key == INGRESS_TOKEN_QUERY_PARAM && !value.trim().is_empty() {
            Some(value.to_string())
        } else {
            None
        }
    })
}

fn sanitized_upstream_path_and_query(uri: &axum::http::Uri) -> String {
    let path = uri.path();
    let Some(query) = uri.query() else {
        return path.to_string();
    };
    let filtered = query
        .split('&')
        .filter(|pair| {
            pair.split_once('=')
                .map(|(key, _)| key != INGRESS_TOKEN_QUERY_PARAM)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>()
        .join("&");
    if filtered.is_empty() {
        path.to_string()
    } else {
        format!("{path}?{filtered}")
    }
}

fn should_skip_request_header(name: &HeaderName) -> bool {
    *name == header::CONNECTION
        || *name == header::TRANSFER_ENCODING
        || *name == header::CONTENT_LENGTH
        || *name == header::AUTHORIZATION
        || *name == header::PROXY_AUTHORIZATION
        || *name == header::COOKIE
}

#[derive(Debug)]
enum PublicIngressError {
    BadRequest(String),
    Unauthorized(String),
    NotFound(String),
    Upstream(anyhow::Error),
}

impl std::fmt::Display for PublicIngressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRequest(message) => write!(f, "{message}"),
            Self::Unauthorized(message) => write!(f, "{message}"),
            Self::NotFound(message) => write!(f, "{message}"),
            Self::Upstream(error) => write!(f, "{error}"),
        }
    }
}

fn normalize_host(host_header: &str) -> String {
    host_header
        .trim()
        .split(':')
        .next()
        .unwrap_or_default()
        .trim()
        .trim_end_matches('.')
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_access_token_from_cookie_reads_common_cookie_name() {
        assert_eq!(
            extract_access_token_from_cookie(Some("foo=bar; access_token=abc123; theme=dark"))
                .as_deref(),
            Some("abc123")
        );
        assert_eq!(
            extract_access_token_from_cookie(Some("accessToken=xyz789")).as_deref(),
            Some("xyz789")
        );
    }

    #[test]
    fn public_ingress_error_maps_unauthorized_to_401() {
        let response = handle_public_ingress_error(PublicIngressError::Unauthorized(
            "unauthorized ingress access".to_string(),
        ));
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn public_ingress_only_routes_running_vms() {
        assert!(vm_state_allows_public_ingress(VmState::Running));
        assert!(!vm_state_allows_public_ingress(VmState::Paused));
        assert!(!vm_state_allows_public_ingress(VmState::Stopped));
        assert!(!vm_state_allows_public_ingress(VmState::Creating));
        assert!(!vm_state_allows_public_ingress(VmState::Error));
    }

    #[test]
    fn public_ingress_denies_sandbox_control_ports() {
        assert!(!public_guest_port_allowed(0));
        assert!(!public_guest_port_allowed(13337));
        assert!(!public_guest_port_allowed(13338));
        assert!(public_guest_port_allowed(3000));
    }

    #[test]
    fn public_ingress_exposures_default_to_auth_required() {
        let exposure = serde_json::from_str::<PublicIngressExposure>(
            r#"{
                "owner_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "public_hostname": "example.test",
                "guest_port": 3000
            }"#,
        )
        .expect("decode exposure");

        assert!(exposure.auth_required);
    }

    fn handle_public_ingress_error(error: PublicIngressError) -> Response {
        let (status, body) = match &error {
            PublicIngressError::BadRequest(message) => (StatusCode::BAD_REQUEST, message.clone()),
            PublicIngressError::Unauthorized(message) => {
                (StatusCode::UNAUTHORIZED, message.clone())
            }
            PublicIngressError::NotFound(message) => (StatusCode::NOT_FOUND, message.clone()),
            PublicIngressError::Upstream(error) => (
                StatusCode::BAD_GATEWAY,
                format!("public ingress proxy failed: {error}"),
            ),
        };
        (status, body).into_response()
    }
}
