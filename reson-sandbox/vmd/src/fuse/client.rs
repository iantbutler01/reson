use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::{Client, StatusCode, header};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize)]
pub struct RemoteDirEntry {
    pub name: String,
    pub kind: String,
    pub size_bytes: u64,
    pub content_hash: Option<String>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RemoteMetadata {
    pub kind: String,
    pub size_bytes: u64,
    pub content_hash: Option<String>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LeaseGrant {
    pub resource_key: String,
    pub owner_token: Uuid,
    pub task_id: Option<Uuid>,
}

#[derive(Clone, Debug, Serialize)]
struct LeaseAcquireRequest<'a> {
    path: &'a str,
    mutation_count: i32,
    component: &'a str,
    run_id: Option<Uuid>,
    reason: &'a str,
}

#[derive(Clone, Debug, Serialize)]
struct LeaseReleaseRequest<'a> {
    resource_key: &'a str,
    owner_token: Uuid,
}

#[derive(Clone, Debug)]
pub struct NymVfsClient {
    client: Client,
    endpoint: String,
    auth_token: String,
    scope_path: String,
}

impl NymVfsClient {
    pub fn new(endpoint: &str, auth_token: &str, scope_path: &str) -> Result<Self> {
        let mut builder = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(90))
            .http2_adaptive_window(true);
        if endpoint.trim_start().starts_with("http://") {
            builder = builder.http2_prior_knowledge();
        }
        let client = builder.build().context("build nymfs reqwest client")?;
        Ok(Self {
            client,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            auth_token: auth_token.to_string(),
            scope_path: scope_path.trim_matches('/').to_string(),
        })
    }

    pub async fn list_dir(&self, path: &str) -> Result<Option<Vec<RemoteDirEntry>>> {
        let response = self
            .request(
                self.client
                    .get(self.url("/tree"))
                    .query(&[("path", self.path_arg(path))]),
            )
            .await?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        response
            .json()
            .await
            .context("decode nymfs tree response")
            .map(Some)
    }

    pub async fn stat(&self, path: &str) -> Result<Option<RemoteMetadata>> {
        let response = self
            .request(
                self.client
                    .get(self.url("/stat"))
                    .query(&[("path", self.path_arg(path))]),
            )
            .await?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            bail!("nymfs stat failed: {}", response.status());
        }
        response
            .json()
            .await
            .context("decode nymfs stat response")
            .map(Some)
    }

    pub async fn read_file_raw(&self, path: &str) -> Result<Vec<u8>> {
        let response = self
            .request(
                self.client
                    .get(self.url("/file/raw"))
                    .query(&[("path", self.path_arg(path))]),
            )
            .await?;
        response
            .bytes()
            .await
            .map(|bytes| bytes.to_vec())
            .context("read nymfs raw bytes")
    }

    pub async fn read_file_range(&self, path: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
        let response = self
            .request(
                self.client
                    .get(self.url("/file/raw"))
                    .query(&[("path", self.path_arg(path))])
                    .header(
                        header::RANGE,
                        format!("bytes={offset}-{}", offset + length.saturating_sub(1)),
                    ),
            )
            .await?;
        response
            .bytes()
            .await
            .map(|bytes| bytes.to_vec())
            .context("read nymfs ranged bytes")
    }

    pub async fn write_file(
        &self,
        path: &str,
        bytes: &[u8],
        lease: &LeaseGrant,
        surface_kind: &str,
        operation: &str,
    ) -> Result<()> {
        self.request(
            self.client
                .put(self.url("/file"))
                .query(&[("path", self.path_arg(path))])
                .header("x-nymfs-component", "vm_runtime")
                .header("x-nymfs-surface-kind", surface_kind)
                .header("x-nymfs-operation", operation)
                .header("x-nymfs-resource-key", lease.resource_key.as_str())
                .header("x-nymfs-lock-owner-token", lease.owner_token.to_string())
                .body(bytes.to_vec()),
        )
        .await?;
        Ok(())
    }

    pub async fn delete_file(
        &self,
        path: &str,
        lease: &LeaseGrant,
        surface_kind: &str,
        operation: &str,
    ) -> Result<()> {
        self.request(
            self.client
                .delete(self.url("/file"))
                .query(&[("path", self.path_arg(path))])
                .header("x-nymfs-component", "vm_runtime")
                .header("x-nymfs-surface-kind", surface_kind)
                .header("x-nymfs-operation", operation)
                .header("x-nymfs-resource-key", lease.resource_key.as_str())
                .header("x-nymfs-lock-owner-token", lease.owner_token.to_string()),
        )
        .await?;
        Ok(())
    }

    pub async fn mkdir(
        &self,
        path: &str,
        lease: &LeaseGrant,
        surface_kind: &str,
        operation: &str,
    ) -> Result<()> {
        self.request(
            self.client
                .put(self.url("/dir"))
                .query(&[("path", self.path_arg(path))])
                .header("x-nymfs-component", "vm_runtime")
                .header("x-nymfs-surface-kind", surface_kind)
                .header("x-nymfs-operation", operation)
                .header("x-nymfs-resource-key", lease.resource_key.as_str())
                .header("x-nymfs-lock-owner-token", lease.owner_token.to_string()),
        )
        .await?;
        Ok(())
    }

    pub async fn rmdir(
        &self,
        path: &str,
        lease: &LeaseGrant,
        surface_kind: &str,
        operation: &str,
    ) -> Result<()> {
        self.request(
            self.client
                .delete(self.url("/dir"))
                .query(&[("path", self.path_arg(path))])
                .header("x-nymfs-component", "vm_runtime")
                .header("x-nymfs-surface-kind", surface_kind)
                .header("x-nymfs-operation", operation)
                .header("x-nymfs-resource-key", lease.resource_key.as_str())
                .header("x-nymfs-lock-owner-token", lease.owner_token.to_string()),
        )
        .await?;
        Ok(())
    }

    pub async fn rename(
        &self,
        from: &str,
        to: &str,
        lease: &LeaseGrant,
        surface_kind: &str,
        operation: &str,
    ) -> Result<()> {
        self.request(
            self.client
                .post(self.url("/rename"))
                .query(&[("from", self.path_arg(from)), ("to", self.path_arg(to))])
                .header("x-nymfs-component", "vm_runtime")
                .header("x-nymfs-surface-kind", surface_kind)
                .header("x-nymfs-operation", operation)
                .header("x-nymfs-resource-key", lease.resource_key.as_str())
                .header("x-nymfs-lock-owner-token", lease.owner_token.to_string()),
        )
        .await?;
        Ok(())
    }

    pub async fn acquire_lease(
        &self,
        path: &str,
        mutation_count: i32,
        reason: &str,
    ) -> Result<LeaseGrant> {
        let scoped_path = self.path_arg(path);
        self.request(
            self.client
                .post(self.url("/lease"))
                .json(&LeaseAcquireRequest {
                    path: scoped_path.as_str(),
                    mutation_count,
                    component: "vm_runtime",
                    run_id: None,
                    reason,
                }),
        )
        .await?
        .json()
        .await
        .context("decode nymfs lease response")
    }

    pub async fn release_lease(&self, lease: &LeaseGrant) -> Result<()> {
        self.request(
            self.client
                .delete(self.url("/lease"))
                .json(&LeaseReleaseRequest {
                    resource_key: lease.resource_key.as_str(),
                    owner_token: lease.owner_token,
                }),
        )
        .await?;
        Ok(())
    }

    fn path_arg(&self, relative: &str) -> String {
        let rel = relative.trim_matches('/');
        if self.scope_path.is_empty() {
            rel.to_string()
        } else if rel.is_empty() {
            self.scope_path.clone()
        } else {
            format!("{}/{}", self.scope_path, rel)
        }
    }

    fn url(&self, suffix: &str) -> String {
        format!("{}{}", self.endpoint, suffix)
    }

    async fn request(&self, builder: reqwest::RequestBuilder) -> Result<reqwest::Response> {
        let response = builder
            .bearer_auth(&self.auth_token)
            .send()
            .await
            .context("send nymfs request")?;
        if response.status().is_success() || response.status() == StatusCode::PARTIAL_CONTENT {
            return Ok(response);
        }
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(response);
        }
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        Err(anyhow!("nymfs request failed: {status} {body}"))
    }
}
