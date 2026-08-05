#!/usr/bin/env bash
# Source this file before starting a headless RoboTwin/SAPIEN client.

_robotwin_find_driver_library() {
  local name="$1"
  local root match
  for root in \
    "${ROBOTWIN_NVIDIA_BUNDLE_DIR:-}" \
    /usr/lib/x86_64-linux-gnu \
    /usr/local/nvidia/lib64 \
    /run/nvidia/driver; do
    [ -d "$root" ] || continue
    match="$(find "$root" -type f -name "$name" -print -quit 2>/dev/null || true)"
    if [ -n "$match" ]; then
      printf '%s\n' "$match"
      return 0
    fi
  done
  return 1
}

if ! ldconfig -p 2>/dev/null | grep 'libvulkan\.so\.1' >/dev/null || \
   ! ldconfig -p 2>/dev/null | grep 'libEGL\.so\.1' >/dev/null; then
  if [ "$(id -u)" -ne 0 ]; then
    echo "RoboTwin renderer requires libvulkan.so.1 and libEGL.so.1; root is needed to install them" >&2
    return 1 2>/dev/null || exit 1
  fi
  command -v flock >/dev/null || {
    echo "RoboTwin renderer setup requires flock to serialize package installation" >&2
    return 1 2>/dev/null || exit 1
  }
  exec 9>/tmp/robotwin-renderer-apt.lock
  flock -w "${ROBOTWIN_RENDERER_APT_LOCK_TIMEOUT:-600}" 9 || {
    echo "Timed out waiting for RoboTwin renderer package installation" >&2
    return 1 2>/dev/null || exit 1
  }
  # Multiple simulator seeds share one job container. Recheck after acquiring
  # the lock so only the first seed mutates the package database.
  if ! ldconfig -p 2>/dev/null | grep 'libvulkan\.so\.1' >/dev/null || \
     ! ldconfig -p 2>/dev/null | grep 'libEGL\.so\.1' >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends libvulkan1 libegl1
    ldconfig
  fi
  flock -u 9
  exec 9>&-
fi

# Volc images can carry NVIDIA userspace libraries older than the host driver.
# CUDA still works in that state, while Vulkan reports INCOMPATIBLE_DRIVER.
driver_version="$(sed -n 's/.*Kernel Module  \([^ ]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -1)"
default_bundle_dir="/vePFS/tim/runtime/nvidia_driver/$driver_version/lib"
if [ -z "${ROBOTWIN_NVIDIA_BUNDLE_DIR:-}" ] && [ -d "$default_bundle_dir" ]; then
  export ROBOTWIN_NVIDIA_BUNDLE_DIR="$default_bundle_dir"
fi
current_glx="$(readlink -f /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null || true)"
if [ -n "$driver_version" ] && \
   { [ -n "${ROBOTWIN_NVIDIA_BUNDLE_DIR:-}" ] || [[ "$current_glx" != *".$driver_version" ]]; }; then
  matching_glx="$(_robotwin_find_driver_library "libGLX_nvidia.so.$driver_version" || true)"
  if [ -z "$matching_glx" ]; then
    echo "Host NVIDIA driver is $driver_version, but its Vulkan userspace libraries are unavailable" >&2
    return 1 2>/dev/null || exit 1
  fi

  driver_lib_dir="/tmp/robotwin-nvidia-$driver_version-$$"
  mkdir -p "$driver_lib_dir"
  required_driver_libraries=(
    libGLX_nvidia libnvidia-glcore libnvidia-glsi libnvidia-tls
    libnvidia-glvkspirv libnvidia-eglcore libEGL_nvidia
  )
  optional_driver_libraries=(libnvidia-vulkan-producer libnvidia-rtcore libnvidia-gpucomp)
  for library_name in "${required_driver_libraries[@]}"; do
    library="$(_robotwin_find_driver_library "$library_name.so.$driver_version" || true)"
    if [ -z "$library" ]; then
      echo "Missing NVIDIA userspace library $library_name.so.$driver_version" >&2
      return 1 2>/dev/null || exit 1
    fi
    ln -sf "$library" "$driver_lib_dir/$(basename "$library")"
  done
  for library_name in "${optional_driver_libraries[@]}"; do
    library="$(_robotwin_find_driver_library "$library_name.so.$driver_version" || true)"
    [ -n "$library" ] || continue
    ln -sf "$library" "$driver_lib_dir/$(basename "$library")"
  done
  ldconfig -n "$driver_lib_dir"

  glx_library="$driver_lib_dir/libGLX_nvidia.so.0"
  if [ ! -e "$glx_library" ]; then
    echo "Failed to construct the matching NVIDIA GLX library for driver $driver_version" >&2
    return 1 2>/dev/null || exit 1
  fi
  unresolved_dependencies=""
  while IFS= read -r library; do
    library_dependencies="$(
      LD_LIBRARY_PATH="$driver_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$library" 2>&1 || true
    )"
    if grep -q 'not found' <<<"$library_dependencies"; then
      unresolved_dependencies+=$'\n'"$library:"$'\n'"$library_dependencies"
    fi
  done < <(find "$driver_lib_dir" -maxdepth 1 -type l -name '*.so.*' -print)
  if [ -n "$unresolved_dependencies" ]; then
    echo "Matching NVIDIA userspace libraries have unresolved dependencies:$unresolved_dependencies" >&2
    return 1 2>/dev/null || exit 1
  fi

  export LD_LIBRARY_PATH="$driver_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  cat > "$driver_lib_dir/nvidia_icd.json.tmp" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "$driver_lib_dir/libGLX_nvidia.so.0",
    "api_version": "1.3.0"
  }
}
EOF
  mv "$driver_lib_dir/nvidia_icd.json.tmp" "$driver_lib_dir/nvidia_icd.json"
  export VK_ICD_FILENAMES="$driver_lib_dir/nvidia_icd.json"

  cat > "$driver_lib_dir/10_nvidia.json.tmp" <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "$driver_lib_dir/libEGL_nvidia.so.0"
  }
}
EOF
  mv "$driver_lib_dir/10_nvidia.json.tmp" "$driver_lib_dir/10_nvidia.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$driver_lib_dir/10_nvidia.json"
else
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi

# Prevent SAPIEN's import-time fallback from selecting an ICD from its shared
# environment instead of the host-matched NVIDIA stack selected above.
if [ -n "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ]; then
  :
elif [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
elif [ -f /etc/glvnd/egl_vendor.d/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=/etc/glvnd/egl_vendor.d/10_nvidia.json
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-runtime}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

unset -f _robotwin_find_driver_library
unset driver_version default_bundle_dir current_glx matching_glx driver_lib_dir glx_library library_dependencies \
  unresolved_dependencies root library library_name required_driver_libraries optional_driver_libraries match
