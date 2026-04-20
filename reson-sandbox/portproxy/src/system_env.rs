// @dive-file: Loads managed guest-wide execution environment defaults for portproxy child processes.
// @dive-rel: Lets bootstrap-installed proxy settings apply to exec, daemon, and interactive shell flows despite env_clear.
// @dive-rel: Request-provided env vars still override these defaults on a per-command basis.
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

const DEFAULT_PROXY_ENV_PATH: &str = "/etc/reson/proxy.env";

pub fn build_exec_env(
    default_path: &str,
    default_home: &str,
    overrides: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut merged = HashMap::new();
    merged.insert("PATH".to_string(), default_path.to_string());
    merged.insert("HOME".to_string(), default_home.to_string());
    merged.extend(read_managed_proxy_env());
    merged.extend(overrides.clone());
    merged
}

fn read_managed_proxy_env() -> HashMap<String, String> {
    let path = managed_proxy_env_path();
    let Ok(contents) = fs::read_to_string(&path) else {
        return HashMap::new();
    };

    contents
        .lines()
        .filter_map(parse_env_line)
        .collect::<HashMap<_, _>>()
}

fn managed_proxy_env_path() -> PathBuf {
    env::var("RESON_PORTPROXY_MANAGED_ENV_FILE")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PROXY_ENV_PATH))
}

fn parse_env_line(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let (key, value) = trimmed.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    Some((key.to_string(), value.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::{LazyLock, Mutex};

    use super::*;

    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[test]
    fn build_exec_env_merges_managed_proxy_env_and_request_overrides() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let tmp = tempfile::tempdir().expect("tempdir");
        let env_file = tmp.path().join("proxy.env");
        fs::write(
            &env_file,
            "http_proxy=http://10.0.2.100:3128\nHTTPS_PROXY=http://10.0.2.100:3128\n# comment\n",
        )
        .expect("write proxy env");

        unsafe {
            env::set_var(
                "RESON_PORTPROXY_MANAGED_ENV_FILE",
                env_file.to_string_lossy().to_string(),
            );
        }

        let overrides = HashMap::from([
            ("PATH".to_string(), "/custom/bin".to_string()),
            ("CUSTOM_FLAG".to_string(), "1".to_string()),
        ]);
        let merged = build_exec_env("/usr/bin", "/root", &overrides);

        assert_eq!(merged.get("PATH").map(String::as_str), Some("/custom/bin"));
        assert_eq!(merged.get("HOME").map(String::as_str), Some("/root"));
        assert_eq!(
            merged.get("http_proxy").map(String::as_str),
            Some("http://10.0.2.100:3128")
        );
        assert_eq!(
            merged.get("HTTPS_PROXY").map(String::as_str),
            Some("http://10.0.2.100:3128")
        );
        assert_eq!(merged.get("CUSTOM_FLAG").map(String::as_str), Some("1"));

        unsafe {
            env::remove_var("RESON_PORTPROXY_MANAGED_ENV_FILE");
        }
    }
}
