#!/usr/bin/env bash
set -euo pipefail

version="${1:?usage: $0 NVIDIA_DRIVER_VERSION [CACHE_ROOT]}"
cache_root="${2:-/vePFS/tim/runtime/nvidia_driver}"
base_url="https://developer.download.nvidia.com/compute/nvidia-driver/redist"
manifest_url="$base_url/redistrib_${version}.json"
archive_name="nvidia_driver-linux-x86_64-${version}-archive.tar.xz"
archive="$cache_root/$archive_name"
bundle_dir="$cache_root/$version/lib"

mkdir -p "$cache_root" "$bundle_dir"
manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
curl -fsSL --retry 3 "$manifest_url" -o "$manifest"

read -r relative_path expected_sha < <(
  python3 - "$manifest" <<'PY'
import json
import sys

entry = json.load(open(sys.argv[1]))["nvidia_driver"]["linux-x86_64"]
print(entry["relative_path"], entry["sha256"])
PY
)

if [ ! -f "$archive" ] || [ "$(sha256sum "$archive" | awk '{print $1}')" != "$expected_sha" ]; then
  rm -f "$archive.part"
  curl -fL --retry 3 "$base_url/$relative_path" -o "$archive.part"
  printf '%s  %s\n' "$expected_sha" "$archive.part" | sha256sum -c -
  mv "$archive.part" "$archive"
fi

archive_root="nvidia_driver-linux-x86_64-${version}-archive"
rm -rf "$bundle_dir"
mkdir -p "$bundle_dir"
tar -xJf "$archive" --strip-components=2 -C "$bundle_dir" "$archive_root/lib"
test -f "$bundle_dir/libGLX_nvidia.so.$version"
test -f "$bundle_dir/libnvidia-glcore.so.$version"
printf 'NVIDIA userspace bundle ready: %s\n' "$bundle_dir"
