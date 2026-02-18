use std::{
    env,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use reqwest::{Client, StatusCode, header::RANGE};
use tokio::{
    fs::{self, OpenOptions},
    io::AsyncWriteExt,
};
use tokio_stream::StreamExt;
use tracing::{debug, info, warn};

use crate::config::{self, Config};

pub const BASE_IMAGE_EXT: &str = ".qcow2";
pub const BASE_IMAGE_SIZE_GB: i32 = 10;
pub const DEFAULT_VM_REGISTRY_URL: &str = "https://vm-images.openbracket.dev";
const MAX_DOWNLOAD_ATTEMPTS: usize = 5;

pub fn base_image_file_name(reference: &str, arch: &str) -> String {
    let sanitized = sanitize_image_reference(reference);
    format!("{sanitized}-{arch}{BASE_IMAGE_EXT}")
}

pub fn default_base_image_path(cfg: &Config, reference: &str, arch: &str) -> PathBuf {
    PathBuf::from(&cfg.data_dir)
        .join(config::BASE_IMAGES_DIR_NAME)
        .join(base_image_file_name(reference, arch))
}

pub fn sanitize_image_reference(reference: &str) -> String {
    reference
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub fn vm_registry_base_url() -> String {
    match env::var("BRACKET_VM_REGISTRY_URL") {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                DEFAULT_VM_REGISTRY_URL.to_string()
            } else {
                trimmed.trim_end_matches('/').to_string()
            }
        }
        Err(_) => DEFAULT_VM_REGISTRY_URL.to_string(),
    }
}

#[derive(Clone, Debug)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub enum PrebuiltImageStatus {
    Downloaded { bytes: u64 },
    NotFound,
}

pub async fn fetch_prebuilt_image(reference: &str, arch: &str, target: &Path) -> Result<bool> {
    match download_prebuilt_image(reference, arch, target).await {
        Ok(PrebuiltImageStatus::Downloaded { .. }) => Ok(true),
        Ok(PrebuiltImageStatus::NotFound) => Ok(false),
        Err(_) => Ok(false),
    }
}

pub async fn download_prebuilt_image(
    reference: &str,
    arch: &str,
    target: &Path,
) -> Result<PrebuiltImageStatus> {
    download_prebuilt_image_with_progress(reference, arch, target, |_| {}).await
}

pub async fn download_prebuilt_image_with_progress<F>(
    reference: &str,
    arch: &str,
    target: &Path,
    mut progress: F,
) -> Result<PrebuiltImageStatus>
where
    F: FnMut(DownloadProgress),
{
    let parent = target
        .parent()
        .context("determine directory for base image download")?;

    let file_name = base_image_file_name(reference, arch);
    let url = format!("{}/{}", vm_registry_base_url(), file_name);
    let client = Client::new();
    debug!(%url, path = %target.display(), "attempting to download prebuilt VM image");

    let tmp_name = format!(
        "{}.part",
        target
            .file_name()
            .and_then(|segment| segment.to_str())
            .unwrap_or("image")
    );
    let tmp_path = parent.join(tmp_name);

    let mut attempt = 0usize;
    let mut last_err = None;

    'attempt: loop {
        if attempt >= MAX_DOWNLOAD_ATTEMPTS {
            break;
        }
        attempt += 1;

        let resume_from = match fs::metadata(&tmp_path).await {
            Ok(metadata) => metadata.len(),
            Err(err) if err.kind() == ErrorKind::NotFound => 0,
            Err(err) => {
                warn!(path = %tmp_path.display(), error = %err, "failed to inspect partial download");
                return Err(err.into());
            }
        };

        let mut request = client.get(&url);
        if resume_from > 0 {
            request = request.header(RANGE, format!("bytes={resume_from}-"));
        }

        let response = match request.send().await {
            Ok(resp) => resp,
            Err(err) => {
                warn!(
                    %url,
                    attempt,
                    error = %err,
                    "unable to contact prebuilt VM image registry"
                );
                last_err = Some(err.into());
                continue;
            }
        };

        match response.status() {
            StatusCode::OK | StatusCode::PARTIAL_CONTENT => {}
            StatusCode::NOT_FOUND => {
                debug!(%url, "no prebuilt VM image available");
                let _ = fs::remove_file(&tmp_path).await;
                return Ok(PrebuiltImageStatus::NotFound);
            }
            status => {
                warn!(
                    %url,
                    status = ?status,
                    "prebuilt VM image registry returned unexpected status"
                );
                return Err(anyhow!("unexpected status {status} from VM registry"));
            }
        }

        let resuming = response.status() == StatusCode::PARTIAL_CONTENT && resume_from > 0;
        if resume_from > 0 && !resuming {
            debug!(
                %url,
                attempt,
                "VM registry did not honor range request; restarting download"
            );
        }
        let mut open_opts = OpenOptions::new();
        open_opts.write(true).create(true);
        if resuming {
            open_opts.append(true);
        } else {
            open_opts.truncate(true);
        }

        let mut file = match open_opts.open(&tmp_path).await {
            Ok(file) => file,
            Err(err) => {
                warn!(path = %tmp_path.display(), error = %err, "cannot open temp file for download");
                return Err(err.into());
            }
        };

        let mut written = if resuming { resume_from } else { 0 };
        let total = response.content_length().map(|length| length + written);
        progress(DownloadProgress {
            downloaded_bytes: written,
            total_bytes: total,
        });

        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(err) => {
                    warn!(
                        attempt,
                        error = %err,
                        "failed to read chunk from remote VM image"
                    );
                    last_err = Some(err.into());
                    continue 'attempt;
                }
            };

            if let Err(err) = file.write_all(&bytes).await {
                warn!(
                    attempt,
                    path = %tmp_path.display(),
                    error = %err,
                    "failed to write remote VM chunk"
                );
                last_err = Some(err.into());
                continue 'attempt;
            }
            written += bytes.len() as u64;
            progress(DownloadProgress {
                downloaded_bytes: written,
                total_bytes: total,
            });
        }

        if let Err(err) = file.flush().await {
            warn!(
                attempt,
                path = %tmp_path.display(),
                error = %err,
                "failed to flush remote VM download"
            );
            last_err = Some(err.into());
            continue 'attempt;
        }

        if let Err(err) = fs::rename(&tmp_path, target).await {
            warn!(
                path = %tmp_path.display(),
                error = %err,
                "failed to move downloaded VM image into place"
            );
            return Err(err.into());
        }

        info!(path = %target.display(), url = %url, "downloaded prebuilt VM image");
        return Ok(PrebuiltImageStatus::Downloaded { bytes: written });
    }

    let err = last_err.unwrap_or_else(|| anyhow!("download attempts exhausted"));
    Err(err.context(format!(
        "failed to download prebuilt VM image after {MAX_DOWNLOAD_ATTEMPTS} attempts"
    )))
}
