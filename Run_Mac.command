#!/bin/bash
# Double-click entry: system tools only until a pinned uv archive is verified.
set -u
COURSE_PAUSE=1
COURSE_OFFLINE=0
COURSE_HELP=0
COURSE_BAD=0
for course_arg in "$@"; do
    case "$course_arg" in
        --no-pause) COURSE_PAUSE=0 ;;
        --offline) COURSE_OFFLINE=1 ;;
        --help|-h) COURSE_HELP=1 ;;
        *) COURSE_BAD=1 ;;
    esac
done
course_finish() {
    course_status=$?
    trap - EXIT
    if [ "$COURSE_PAUSE" -eq 1 ]; then
        printf '\nExit code: %s. Press Return to close.\n' "$course_status"
        read -r _ || true
    fi
    exit "$course_status"
}
trap course_finish EXIT
course_fail() { printf 'COURSE ERROR %s: %s\n' "$1" "$2" >&2; exit "$1"; }
[ "$COURSE_BAD" -eq 0 ] || course_fail 2 'Unknown option. Use --offline, --no-pause, or --help.'
if [ "$COURSE_HELP" -eq 1 ]; then
    printf 'AFQMC course launcher: Run_Mac.command [--offline] [--no-pause]\nFirst run configures private Python 3.12.10; later runs can be offline.\n'
    exit 0
fi
[ "$(/usr/bin/uname -s)" = Darwin ] && [ "$(/usr/bin/uname -m)" = arm64 ] || course_fail 14 'Requires native macOS ARM64 (Apple Silicon).'
course_macos=$(/usr/bin/sw_vers -productVersion)
[ "${course_macos%%.*}" -ge 14 ] || course_fail 14 'Requires macOS 14 or newer.'
COURSE_ROOT=$(cd "$(dirname "$0")" && pwd -P) || course_fail 12 'Cannot open project directory.'
COURSE_RUNTIME="$COURSE_ROOT/.course_runtime"
course_safe() {
    course_path="$1"
    while [ "$course_path" != / ]; do
        if [ -L "$course_path" ]; then
            case "$course_path" in /tmp|/var) ;; *) course_fail 12 'Symlink runtime/output paths are not accepted; existing files preserved.' ;; esac
        fi
        course_path=$(dirname "$course_path")
    done
}
for course_path in "$COURSE_RUNTIME" "$COURSE_RUNTIME/bootstrap" "$COURSE_RUNTIME/cache" "$COURSE_RUNTIME/python" "$COURSE_RUNTIME/envs" "$COURSE_ROOT/course_validation"; do course_safe "$course_path"; done
mkdir -p "$COURSE_RUNTIME/bootstrap" || course_fail 12 'Project must be writable.'
COURSE_ARCHIVE="$COURSE_RUNTIME/uv-0.13.5-macos-arm64.tar.gz"
COURSE_SHA=743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73
course_safe "$COURSE_ARCHIVE"
course_hash() { /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'; }
if [ ! -f "$COURSE_ARCHIVE" ]; then
    [ "$COURSE_OFFLINE" -eq 0 ] || course_fail 10 'OFFLINE: verified uv cache is missing. Run once online in this extracted directory.'
    COURSE_PART=$(/usr/bin/mktemp "$COURSE_RUNTIME/uv-download.XXXXXX") || course_fail 12 'Cannot create download file.'
    printf 'Downloading official uv 0.13.5 (first setup)...\n'
    /usr/bin/curl --fail --location --proto '=https' --proto-redir '=https' --connect-timeout 20 --max-time 180 --retry 2 --output "$COURSE_PART" 'https://github.com/astral-sh/uv/releases/download/0.13.5/uv-aarch64-apple-darwin.tar.gz' || course_fail 13 'uv download failed. Check connection and retry; existing files preserved.'
    [ "$(course_hash "$COURSE_PART")" = "$COURSE_SHA" ] || course_fail 11 'Downloaded uv SHA-256 mismatch; untrusted bytes were not executed.'
    /bin/ln "$COURSE_PART" "$COURSE_ARCHIVE" || course_fail 12 'uv cache appeared concurrently; retry. Existing files preserved.'
    /bin/rm "$COURSE_PART"
fi
[ "$(course_hash "$COURSE_ARCHIVE")" = "$COURSE_SHA" ] || course_fail 11 'Cached uv SHA-256 mismatch; cache preserved and never executed. Restore a verified cache or use a new extraction.'
# Extract the executable anew from the checked archive: never trust an old uv binary.
COURSE_BOOT=$(/usr/bin/mktemp -d "$COURSE_RUNTIME/bootstrap/run.XXXXXX") || course_fail 12 'Cannot create bootstrap directory.'
/usr/bin/tar -xzf "$COURSE_ARCHIVE" -C "$COURSE_BOOT" uv-aarch64-apple-darwin/uv || course_fail 11 'Verified archive could not be extracted.'
COURSE_UV="$COURSE_BOOT/uv-aarch64-apple-darwin/uv"
[ -f "$COURSE_UV" ] && [ ! -L "$COURSE_UV" ] || course_fail 11 'uv executable is missing or unsafe.'
# Ignore outside uv/Python configuration, without changing HOME, PATH or user files.
for course_key in $(/usr/bin/env | /usr/bin/cut -d= -f1); do
    case "$course_key" in UV_*|PYTHON*|PIP_*|VIRTUAL_ENV|CONDA_PREFIX) unset "$course_key" ;; esac
done
export UV_NO_CONFIG=1 UV_NO_PROGRESS=1 UV_PYTHON_INSTALL_BIN=0 UV_PYTHON_INSTALL_REGISTRY=0 UV_PYTHON_NO_REGISTRY=1
export UV_CACHE_DIR="$COURSE_RUNTIME/cache" UV_PYTHON_INSTALL_DIR="$COURSE_RUNTIME/python"
export UV_PYTHON_BIN_DIR="$COURSE_RUNTIME/bin" UV_TOOL_DIR="$COURSE_RUNTIME/tools" UV_TOOL_BIN_DIR="$COURSE_RUNTIME/bin"
export UV_DEFAULT_INDEX=https://pypi.org/simple UV_PYTHON_PREFERENCE=only-managed
export UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=2
if [ "$COURSE_OFFLINE" -eq 1 ]; then export UV_OFFLINE=1; fi
cd "$COURSE_ROOT" || course_fail 12 'Cannot enter project.'
printf 'Checking private CPython 3.12.10...\n'
"$COURSE_UV" run --no-project --no-config --python 3.12.10 -- python -I -X utf8 "$COURSE_ROOT/scripts/start_course.py" --uv "$COURSE_UV" "$@"
COURSE_RESULT=$?
# Only the fresh extraction is removed. The verified archive and Python are retained.
/bin/rm "$COURSE_UV" 2>/dev/null || true
/bin/rmdir "$COURSE_BOOT/uv-aarch64-apple-darwin" "$COURSE_BOOT" 2>/dev/null || true
if [ "$COURSE_RESULT" -ne 0 ]; then
    printf 'Launcher failed (exit %s). If Python setup failed, check network/cache and retry. Existing results are preserved.\n' "$COURSE_RESULT" >&2
fi
exit "$COURSE_RESULT"
