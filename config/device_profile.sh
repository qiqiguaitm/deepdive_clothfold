#!/usr/bin/env bash
# Resolve and export one flat config/device_profiles/<hostname>.yml.

resolve_kai0_device_profile() {
    local root="$1" host candidate
    host="${KAI0_MACHINE_ID:-$(hostname -s 2>/dev/null || echo unknown)}"
    candidate="${KAI0_DEVICE_PROFILE:-$root/config/device_profiles/$host.yml}"
    [[ -f "$candidate" ]] && printf '%s\n' "$candidate"
}

_kai0_profile_value() {
    local file="$1" key="$2"
    sed -nE "s/^[[:space:]]*${key}:[[:space:]]*[\"']?([^\"'#]+)[\"']?[[:space:]]*(#.*)?$/\\1/p" "$file" \
        | head -n1 | sed -E 's/[[:space:]]+$//'
}

load_kai0_device_profile() {
    local root="$1" file value
    file="$(resolve_kai0_device_profile "$root")"
    [[ -n "$file" ]] || return 0
    export KAI0_DEVICE_PROFILE_PATH="$file"

    value="$(_kai0_profile_value "$file" machine_id)"
    [[ -n "$value" ]] && export KAI0_MACHINE_ID="$value"
    value="$(_kai0_profile_value "$file" dataset_chunk)"
    if [[ -n "$value" ]]; then
        [[ "$value" =~ ^[0-9]+$ ]] || {
            echo "[FAIL] invalid dataset_chunk '$value' in $file" >&2
            return 2
        }
        export KAI0_DATASET_CHUNK="$((10#$value))"
    fi

    local role env_key
    for role in top_head hand_left hand_right mid_head; do
        env_key="$(printf '%s' "$role" | tr '[:lower:]' '[:upper:]')"
        value="$(_kai0_profile_value "$file" "camera_${role}_serial")"
        [[ -n "$value" ]] && export "KAI0_CAMERA_${env_key}_SERIAL=$value"
    done
    value="$(_kai0_profile_value "$file" camera_mid_head_enabled)"
    [[ -n "$value" ]] && export KAI0_ENABLE_MID_HEAD="$value"
    value="$(_kai0_profile_value "$file" camera_mid_head_type)"
    [[ -n "$value" ]] && export KAI0_MID_HEAD_TYPE="$value"
    value="$(_kai0_profile_value "$file" camera_mid_head_device)"
    [[ -n "$value" ]] && export KAI0_CAMERA_MID_HEAD_DEVICE="$value"
}
