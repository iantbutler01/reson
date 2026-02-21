#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

GRACE_SECONDS="${LEAK_GRACE_SECONDS:-3}"
INCLUDE_VMD="${LEAK_CHECK_INCLUDE_VMD:-0}"

usage() {
  cat <<'EOF'
Usage: verify_no_leaks.sh [--grace-seconds N] -- <command> [args...]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grace-seconds)
      GRACE_SECONDS="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      err "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  err "no command provided to leak harness"
  usage
  exit 2
fi

count_pattern() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    echo "0"
  else
    printf '%s\n' "$pids" | wc -l | tr -d ' '
  fi
}

before_qemu="$(count_pattern 'qemu-system')"
before_portproxy="$(count_pattern '(^|/)portproxy($|[-_ ])|portproxy-')"
before_vmd="0"
if [[ "$INCLUDE_VMD" == "1" ]]; then
  before_vmd="$(count_pattern '(^|/)vmd($| )| vmd --listen ')"
fi

if [[ "$INCLUDE_VMD" == "1" ]]; then
  log "leak harness baseline: qemu=$before_qemu portproxy=$before_portproxy vmd=$before_vmd"
else
  log "leak harness baseline: qemu=$before_qemu portproxy=$before_portproxy"
fi
log "running command under leak harness: $*"
"$@"

sleep "$GRACE_SECONDS"

after_qemu="$(count_pattern 'qemu-system')"
after_portproxy="$(count_pattern '(^|/)portproxy($|[-_ ])|portproxy-')"
after_vmd="0"
if [[ "$INCLUDE_VMD" == "1" ]]; then
  after_vmd="$(count_pattern '(^|/)vmd($| )| vmd --listen ')"
fi

if [[ "$INCLUDE_VMD" == "1" ]]; then
  log "leak harness after: qemu=$after_qemu portproxy=$after_portproxy vmd=$after_vmd"
else
  log "leak harness after: qemu=$after_qemu portproxy=$after_portproxy"
fi

leak=0
if (( after_qemu > before_qemu )); then
  err "qemu process leak detected: before=$before_qemu after=$after_qemu"
  leak=1
fi
if (( after_portproxy > before_portproxy )); then
  err "portproxy process leak detected: before=$before_portproxy after=$after_portproxy"
  leak=1
fi
if [[ "$INCLUDE_VMD" == "1" ]] && (( after_vmd > before_vmd )); then
  err "vmd process leak detected: before=$before_vmd after=$after_vmd"
  leak=1
fi

if (( leak == 1 )); then
  exit 1
fi

log "no process leaks detected"
