use std::fs;
use std::io::{Seek, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc};

use crate::assets::portproxy;

const INIT_SCRIPT_NAME: &str = "init.sh";
const PORTPROXY_NAME: &str = "portproxy";
const VOLUME_ID: &str = "brkboot";
const SECTOR_SIZE: u32 = 2048;

#[derive(Clone, Debug)]
pub struct Config {
    pub instance_id: String,
    pub hostname: String,
    pub arch: String,
}

pub fn create_iso<P: AsRef<Path>>(path: P, cfg: Config) -> Result<()> {
    if cfg.instance_id.trim().is_empty() {
        bail!("bootstrap iso: instance ID required");
    }
    let hostname = if cfg.hostname.trim().is_empty() {
        cfg.instance_id.clone()
    } else {
        cfg.hostname.clone()
    };

    let bin = portproxy::binary(cfg.arch.as_str())
        .with_context(|| "bootstrap iso: locate portproxy binary")?;
    let init = build_init_script(&hostname);

    let entries = vec![
        IsoEntry::new(INIT_SCRIPT_NAME, init.into_bytes()),
        IsoEntry::new(PORTPROXY_NAME, bin),
    ];

    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("bootstrap iso: ensure directory {}", parent.display()))?;
        }
    }

    write_iso(path.as_ref(), VOLUME_ID, &entries)
}

fn build_init_script(hostname: &str) -> String {
    let host_value = shell_escape(hostname);
    format!(
        r#"#!/bin/bash
set -euxo pipefail

HOSTNAME={host_value}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -n "$HOSTNAME" ]; then
  if command -v hostnamectl >/dev/null 2>&1; then
    hostnamectl set-hostname --static "$HOSTNAME" || true
    hostnamectl set-hostname --pretty "$HOSTNAME" || true
  else
    echo "$HOSTNAME" >/etc/hostname
  fi
  if [ -w /etc/hosts ]; then
    if grep -q "127\.0\.1\.1" /etc/hosts; then
      sed -i "s/^127\.0\.1\.1.*/127.0.1.1 ${{HOSTNAME}}/g" /etc/hosts || true
    else
      printf '127.0.1.1 %s\n' "$HOSTNAME" >> /etc/hosts
    fi
  fi
fi

SRC="$SCRIPT_DIR/portproxy"
DEST="/usr/sbin/portproxy"
if [ ! -f "$SRC" ]; then
  echo "portproxy binary missing on bootstrap volume" >&2
  exit 1
fi
install -m 0755 "$SRC" "$DEST"

cat <<'EOF' >/etc/systemd/system/portproxy.service
[Unit]
Description=Bracket PortProxy
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/sbin/portproxy --server
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat <<'EOF' >/usr/local/sbin/firstboot-resize-rootfs.sh
#!/bin/bash
set -eux

DONE_MARKER=/var/lib/bracket/firstboot-resize-rootfs.done
mkdir -p "$(dirname "$DONE_MARKER")"
if [ -f "$DONE_MARKER" ]; then
  exit 0
fi

ROOT_DEV=$(findmnt -n -o SOURCE / || true)
if [ -z "$ROOT_DEV" ]; then
  exit 0
fi

DISK=""
PART=""
PARENT=$(lsblk -no PKNAME "$ROOT_DEV" 2>/dev/null || true)
PARTNUM=$(lsblk -no PARTNUM "$ROOT_DEV" 2>/dev/null || true)
if [ -n "$PARENT" ] && [ -n "$PARTNUM" ]; then
  DISK="/dev/$PARENT"
  PART="$PARTNUM"
else
  case "$ROOT_DEV" in
    /dev/nvme*|/dev/mmcblk*)
      DISK="${{ROOT_DEV%p*}}"
      PART="${{ROOT_DEV##*p}}"
      ;;
    *)
      DISK="${{ROOT_DEV%[0-9]*}}"
      PART="${{ROOT_DEV#$DISK}}"
      ;;
  esac
fi

if [ -n "$DISK" ] && [ -n "$PART" ]; then
  if ! command -v growpart >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update || true
      apt-get install -y cloud-guest-utils || true
    fi
  fi
  if command -v growpart >/dev/null 2>&1; then
    growpart "$DISK" "$PART" || true
  fi
  if command -v resize2fs >/dev/null 2>&1; then
    resize2fs "$ROOT_DEV" || true
  fi
fi

touch "$DONE_MARKER"
EOF
chmod 0755 /usr/local/sbin/firstboot-resize-rootfs.sh

cat <<'EOF' >/etc/systemd/system/firstboot-resize-rootfs.service
[Unit]
Description=Resize root filesystem on first boot
ConditionPathExists=!/var/lib/bracket/firstboot-resize-rootfs.done

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/firstboot-resize-rootfs.sh

[Install]
WantedBy=multi-user.target
EOF

cat <<'EOF' >/etc/systemd/system/firstboot-resize-rootfs.timer
[Unit]
Description=Run first boot root filesystem resize

[Timer]
OnBootSec=30s
Unit=firstboot-resize-rootfs.service
Persistent=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable portproxy.service
systemctl start portproxy.service
systemctl enable firstboot-resize-rootfs.timer
systemctl start firstboot-resize-rootfs.timer
"#
    )
}

fn shell_escape(value: &str) -> String {
    if value.is_empty() {
        "''".to_string()
    } else if !value.contains('\'') {
        format!("'{value}'")
    } else {
        let mut out = String::with_capacity(value.len() + 2);
        out.push('\'');
        for ch in value.chars() {
            if ch == '\'' {
                out.push_str("'\\''");
            } else {
                out.push(ch);
            }
        }
        out.push('\'');
        out
    }
}

#[derive(Clone)]
struct IsoEntry {
    name: String,
    data: Vec<u8>,
}

impl IsoEntry {
    fn new(name: &str, data: Vec<u8>) -> Self {
        Self {
            name: name.to_string(),
            data,
        }
    }
}

struct LayoutEntry {
    _name: String,
    data: Vec<u8>,
    ident: String,
    extent: u32,
}

struct IsoLayout {
    entries: Vec<LayoutEntry>,
    root_dir_sector: u32,
    root_dir_data: Vec<u8>,
    root_dir_size: usize,
    l_path_table: Vec<u8>,
    m_path_table: Vec<u8>,
    l_path_table_sector: u32,
    m_path_table_sector: u32,
    total_sectors: u32,
}

fn write_iso(path: &Path, volume_id: &str, files: &[IsoEntry]) -> Result<()> {
    if files.is_empty() {
        bail!("iso: at least one file required");
    }

    let now = current_time();
    let layout = prepare_layout(files, now)?;

    let mut file = fs::File::create(path)?;
    file.set_len((layout.total_sectors as u64) * (SECTOR_SIZE as u64))?;

    write_sector(
        &mut file,
        16,
        &build_primary_volume_descriptor(volume_id, &layout, now),
    )?;
    write_sector(&mut file, 17, &build_terminator_descriptor())?;
    write_data(&mut file, layout.l_path_table_sector, &layout.l_path_table)?;
    write_data(&mut file, layout.m_path_table_sector, &layout.m_path_table)?;
    write_data(&mut file, layout.root_dir_sector, &layout.root_dir_data)?;
    for entry in &layout.entries {
        write_data(&mut file, entry.extent, &entry.data)?;
    }

    Ok(())
}

fn prepare_layout(files: &[IsoEntry], ts: DateTime<Utc>) -> Result<IsoLayout> {
    let mut entries = Vec::with_capacity(files.len());
    for file in files {
        entries.push(LayoutEntry {
            _name: file.name.clone(),
            data: file.data.clone(),
            ident: make_identifier(&file.name),
            extent: 0,
        });
    }

    let dir_size = directory_data_length(&entries);
    let root_dir_sectors = sectors_for(dir_size);
    let l_path_sector = 18;
    let m_path_sector = 19;
    let mut current = 20 + root_dir_sectors as u32;
    for entry in &mut entries {
        entry.extent = current;
        current += sectors_for(entry.data.len()) as u32;
    }

    let total_sectors = current;
    let l_path = build_path_table(true, 20);
    let m_path = build_path_table(false, 20);
    let root_dir = build_root_directory(&entries, 20, dir_size, ts)?;

    Ok(IsoLayout {
        entries,
        root_dir_sector: 20,
        root_dir_data: root_dir.clone(),
        root_dir_size: root_dir.len(),
        l_path_table: l_path,
        m_path_table: m_path,
        l_path_table_sector: l_path_sector,
        m_path_table_sector: m_path_sector,
        total_sectors,
    })
}

fn write_sector(file: &mut fs::File, sector: u32, data: &[u8]) -> Result<()> {
    if data.len() != SECTOR_SIZE as usize {
        bail!("iso: sector {sector} invalid size {}", data.len());
    }
    file.seek(std::io::SeekFrom::Start(sector as u64 * SECTOR_SIZE as u64))?;
    file.write_all(data)?;
    Ok(())
}

fn write_data(file: &mut fs::File, sector: u32, data: &[u8]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    file.seek(std::io::SeekFrom::Start(sector as u64 * SECTOR_SIZE as u64))?;
    file.write_all(&pad(data))?;
    Ok(())
}

fn build_primary_volume_descriptor(
    volume_id: &str,
    layout: &IsoLayout,
    ts: DateTime<Utc>,
) -> Vec<u8> {
    let mut buf = vec![0u8; SECTOR_SIZE as usize];
    buf[0] = 1;
    buf[1..6].copy_from_slice(b"CD001");
    buf[6] = 1;
    write_string(&mut buf[8..40], "GO-ISO");
    write_string(&mut buf[40..72], &volume_id.to_uppercase());
    put_both32(&mut buf[80..88], layout.total_sectors);
    put_both16(&mut buf[120..124], 1);
    put_both16(&mut buf[124..128], 1);
    put_both16(&mut buf[128..132], SECTOR_SIZE as u16);
    put_both32(&mut buf[132..140], layout.l_path_table.len() as u32);
    buf[140..144].copy_from_slice(&(layout.l_path_table_sector as u32).to_le_bytes());
    buf[144..148].copy_from_slice(&0u32.to_le_bytes());
    buf[148..152].copy_from_slice(&(layout.m_path_table_sector as u32).to_be_bytes());
    buf[152..156].copy_from_slice(&0u32.to_be_bytes());

    let root_record = build_directory_record(
        layout.root_dir_sector,
        layout.root_dir_size as u32,
        0x02,
        &[0],
        ts,
    );
    buf[156..156 + root_record.len()].copy_from_slice(&root_record);

    write_volume_timestamp(&mut buf[813..830], ts);
    write_volume_timestamp(&mut buf[830..847], ts);
    write_volume_timestamp(&mut buf[847..864], ts);
    write_volume_timestamp(&mut buf[864..881], ts);
    buf[881] = 0x01;
    buf
}

fn build_terminator_descriptor() -> Vec<u8> {
    let mut buf = vec![0u8; SECTOR_SIZE as usize];
    buf[0] = 255;
    buf[1..6].copy_from_slice(b"CD001");
    buf[6] = 1;
    buf
}

fn build_path_table(little_endian: bool, root_sector: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.push(1);
    buf.push(0);
    if little_endian {
        buf.extend_from_slice(&root_sector.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
    } else {
        buf.extend_from_slice(&root_sector.to_be_bytes());
        buf.extend_from_slice(&1u16.to_be_bytes());
    }
    buf.push(0);
    buf.push(0);
    pad(&buf)
}

fn build_root_directory(
    entries: &[LayoutEntry],
    sector: u32,
    dir_size: usize,
    ts: DateTime<Utc>,
) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(dir_size);
    buf.extend_from_slice(&build_directory_record(
        sector,
        dir_size as u32,
        0x02,
        &[0],
        ts,
    ));
    buf.extend_from_slice(&build_directory_record(
        sector,
        dir_size as u32,
        0x02,
        &[1],
        ts,
    ));
    for entry in entries {
        buf.extend_from_slice(&build_directory_record(
            entry.extent,
            entry.data.len() as u32,
            0x00,
            entry.ident.as_bytes(),
            ts,
        ));
    }
    Ok(pad(&buf))
}

fn build_directory_record(
    extent: u32,
    size: u32,
    flags: u8,
    ident: &[u8],
    ts: DateTime<Utc>,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(33 + ident.len());
    buf.push(33 + ident.len() as u8 + (ident.len() % 2 == 0) as u8);
    buf.push(0);
    buf.extend_from_slice(&extent.to_le_bytes());
    buf.extend_from_slice(&extent.to_be_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(&datetime_to_record(ts));
    buf.push(flags);
    buf.push(0); // file unit size
    buf.push(0); // interleave gap size
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_be_bytes());
    buf.push(ident.len() as u8);
    buf.extend_from_slice(ident);
    if ident.len() % 2 == 0 {
        buf.push(0);
    }
    buf
}

fn datetime_to_record(ts: DateTime<Utc>) -> [u8; 7] {
    [
        (ts.year() - 1900) as u8,
        ts.month() as u8,
        ts.day() as u8,
        ts.hour() as u8,
        ts.minute() as u8,
        ts.second() as u8,
        0,
    ]
}

fn put_both16(buf: &mut [u8], value: u16) {
    buf[..2].copy_from_slice(&value.to_le_bytes());
    buf[2..4].copy_from_slice(&value.to_be_bytes());
}

fn put_both32(buf: &mut [u8], value: u32) {
    buf[..4].copy_from_slice(&value.to_le_bytes());
    buf[4..8].copy_from_slice(&value.to_be_bytes());
}

fn write_string(target: &mut [u8], value: &str) {
    for (dst, src) in target.iter_mut().zip(value.bytes()) {
        *dst = src;
    }
}

fn write_volume_timestamp(buf: &mut [u8], ts: DateTime<Utc>) {
    let formatted = ts.format("%Y%m%d%H%M%S00").to_string();
    write_string(buf, &formatted);
}

fn directory_data_length(entries: &[LayoutEntry]) -> usize {
    let mut size = 0;
    size += build_directory_record(0, 0, 0x02, &[0], Utc::now()).len();
    size += build_directory_record(0, 0, 0x02, &[1], Utc::now()).len();
    for entry in entries {
        size += build_directory_record(
            entry.extent,
            entry.data.len() as u32,
            0,
            entry.ident.as_bytes(),
            Utc::now(),
        )
        .len();
    }
    size
}

fn sectors_for(size: usize) -> usize {
    (size + SECTOR_SIZE as usize - 1) / SECTOR_SIZE as usize
}

fn pad(data: &[u8]) -> Vec<u8> {
    let sectors = sectors_for(data.len());
    let mut out = vec![0u8; sectors * SECTOR_SIZE as usize];
    out[..data.len()].copy_from_slice(data);
    out
}

fn make_identifier(name: &str) -> String {
    format!("{};1", name.to_uppercase())
}

fn current_time() -> DateTime<Utc> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards");
    Utc.timestamp_opt(now.as_secs() as i64, now.subsec_nanos())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::ensure;
    use std::fs;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn seed_iso_contains_required_files() -> Result<()> {
        let proxy_dir = tempdir()?;
        let binary_path = proxy_dir.path().join("portproxy-linux-amd64");
        fs::write(&binary_path, b"test-portproxy")?;
        unsafe {
            std::env::set_var("PROXY_BIN", proxy_dir.path());
        }

        let tmp_dir = tempdir()?;
        let iso_path = tmp_dir.path().join("bootstrap.iso");
        let cfg = Config {
            instance_id: "vm-test".to_string(),
            hostname: "vm-test".to_string(),
            arch: "amd64".to_string(),
        };
        create_iso(&iso_path, cfg)?;

        let entries = read_root_entries(&iso_path)?;
        assert!(
            entries.contains(&"INIT.SH;1".to_string()),
            "init script missing"
        );
        assert!(
            entries.contains(&"PORTPROXY;1".to_string()),
            "portproxy missing"
        );
        Ok(())
    }

    fn read_root_entries(path: &Path) -> Result<Vec<String>> {
        let mut file = fs::File::open(path)?;
        let root_record = read_root_directory_record(&mut file)?;
        let mut buf = vec![0u8; root_record.size as usize];
        file.seek(SeekFrom::Start(
            root_record.extent as u64 * SECTOR_SIZE as u64,
        ))?;
        file.read_exact(&mut buf)?;

        let mut entries = Vec::new();
        let mut offset = 0usize;
        while offset < buf.len() {
            if buf[offset] == 0 {
                let next_sector = ((offset / SECTOR_SIZE as usize) + 1) * SECTOR_SIZE as usize;
                if next_sector <= offset {
                    break;
                }
                offset = next_sector;
                continue;
            }

            let length = buf[offset] as usize;
            if offset + length > buf.len() || length < 34 {
                break;
            }
            let ident_len = buf[offset + 32] as usize;
            if 33 + ident_len > length {
                break;
            }
            let ident = &buf[offset + 33..offset + 33 + ident_len];
            if ident != [0] && ident != [1] {
                entries.push(String::from_utf8_lossy(ident).to_string());
            }
            offset += length;
        }
        Ok(entries)
    }

    struct DirectoryRecord {
        extent: u32,
        size: u32,
    }

    fn read_root_directory_record(file: &mut fs::File) -> Result<DirectoryRecord> {
        file.seek(SeekFrom::Start(SECTOR_SIZE as u64 * 16))?;
        let mut sector = vec![0u8; SECTOR_SIZE as usize];
        file.read_exact(&mut sector)?;
        ensure!(sector[0] == 1, "primary volume descriptor missing");
        let record_len = sector[156] as usize;
        ensure!(record_len >= 34, "invalid root directory record");
        let record = &sector[156..156 + record_len];
        let extent = u32::from_le_bytes([record[2], record[3], record[4], record[5]]);
        let size = u32::from_le_bytes([record[10], record[11], record[12], record[13]]);
        Ok(DirectoryRecord { extent, size })
    }
}
