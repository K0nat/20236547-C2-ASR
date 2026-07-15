#!/usr/bin/env bash

set -uo pipefail

MODEL_DIR="${MODEL_DIR:-/root/siton-tmp/multimodal/c2/models}"
LOG_DIR="${LOG_DIR:-/root/siton-tmp/multimodal/c2/logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
INSTALL_MODELSCOPE="${INSTALL_MODELSCOPE:-1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

MODELS=(
  "openai/whisper-large-v3"
  "openai/whisper-large-v3-turbo"
)

INCLUDE_PATTERNS=(
  "*.json"
  "*.txt"
  "vocab.json"
  "merges.txt"
  "tokenizer.model"
  "tokenizer.json"
  "model.safetensors"
  "model.safetensors.index.json"
  "model-*.safetensors"
  "generation_config.json"
  "preprocessor_config.json"
)

EXCLUDE_PATTERNS=(
  "*.bin"
  "*.h5"
  "*.onnx"
  "*.ot"
  "*.msgpack"
  "*fp16*"
  "*fp32*"
  "flax_model.*"
  "pytorch_model.*"
  "tf_model.*"
)

mkdir -p "$MODEL_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/download_whisper_models_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

declare -A MODEL_STATUS
declare -A MODEL_SOURCE
declare -A MODEL_PATH

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

model_name() {
  basename "$1"
}

target_dir_for() {
  printf '%s/%s\n' "$MODEL_DIR" "$(model_name "$1")"
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

ensure_modelscope() {
  if have_command modelscope; then
    return 0
  fi

  if python3 -c 'import modelscope' >/dev/null 2>&1; then
    return 0
  fi

  if [[ "$INSTALL_MODELSCOPE" != "1" ]]; then
    log "ModelScope is not installed, and INSTALL_MODELSCOPE=$INSTALL_MODELSCOPE. Skip ModelScope fallback."
    return 1
  fi

  log "ModelScope is not installed. Installing with pip index: $PIP_INDEX_URL"
  python3 -m pip install -U modelscope -i "$PIP_INDEX_URL"
}

pattern_args_repeatable() {
  local p
  for p in "${INCLUDE_PATTERNS[@]}"; do
    printf '%s\0%s\0' --include "$p"
  done
  for p in "${EXCLUDE_PATTERNS[@]}"; do
    printf '%s\0%s\0' --exclude "$p"
  done
}

download_hf_mirror() {
  local repo_id="$1"
  local target_dir="$2"
  local args=()

  log "Trying model: $repo_id"
  log "Source: HF-Mirror"
  log "HF_ENDPOINT: $HF_ENDPOINT"
  log "Target directory: $target_dir"
  log "Include patterns: ${INCLUDE_PATTERNS[*]}"
  log "Exclude patterns: ${EXCLUDE_PATTERNS[*]}"
  mkdir -p "$target_dir"

  while IFS= read -r -d '' arg; do
    args+=("$arg")
  done < <(pattern_args_repeatable)

  if have_command hf; then
    HF_ENDPOINT="$HF_ENDPOINT" hf download "$repo_id" \
      "${args[@]}" \
      --local-dir "$target_dir"
    return $?
  fi

  if have_command huggingface-cli; then
    HF_ENDPOINT="$HF_ENDPOINT" huggingface-cli download "$repo_id" \
      "${args[@]}" \
      --local-dir "$target_dir" \
      --local-dir-use-symlinks False \
      --resume-download
    return $?
  fi

  log "Neither hf nor huggingface-cli was found."
  return 127
}

modelscope_candidates() {
  local repo_id="$1"
  local short_name
  short_name="$(model_name "$repo_id")"

  printf '%s\n' \
    "openai/$short_name" \
    "AI-ModelScope/$short_name" \
    "iic/$short_name"
}

download_modelscope_one() {
  local ms_id="$1"
  local target_dir="$2"
  local args=()

  log "Trying ModelScope model id: $ms_id"
  log "Source: ModelScope"
  log "Target directory: $target_dir"
  log "Include patterns: ${INCLUDE_PATTERNS[*]}"
  log "Exclude patterns: ${EXCLUDE_PATTERNS[*]}"
  mkdir -p "$target_dir"

  args+=(--include "${INCLUDE_PATTERNS[@]}")
  args+=(--exclude "${EXCLUDE_PATTERNS[@]}")
  args+=(--max-workers 4)

  if have_command modelscope; then
    modelscope download "$ms_id" --local-dir "$target_dir" "${args[@]}"
    return $?
  fi

  python3 -m modelscope.cli.cli download "$ms_id" --local-dir "$target_dir" "${args[@]}"
}

download_modelscope() {
  local repo_id="$1"
  local target_dir="$2"
  local ms_id

  ensure_modelscope || return 1

  while IFS= read -r ms_id; do
    if download_modelscope_one "$ms_id" "$target_dir"; then
      log "ModelScope download command succeeded for $ms_id."
      return 0
    fi
    log "ModelScope download command failed for $ms_id. Continue with next candidate."
  done < <(modelscope_candidates "$repo_id")

  return 1
}

has_required_files() {
  local target_dir="$1"
  local missing=0
  local size

  for f in config.json preprocessor_config.json tokenizer.json; do
    if [[ -s "$target_dir/$f" ]]; then
      log "Check OK: $target_dir/$f"
    else
      log "Check MISSING: $target_dir/$f"
      missing=1
    fi
  done

  if [[ -s "$target_dir/model.safetensors" ]]; then
    size="$(stat -c '%s' "$target_dir/model.safetensors" 2>/dev/null || echo 0)"
    if (( size > 104857600 )); then
      log "Check OK: $target_dir/model.safetensors (${size} bytes)"
    else
      log "Check TOO SMALL: $target_dir/model.safetensors (${size} bytes)"
      missing=1
    fi
  elif [[ -s "$target_dir/model.safetensors.index.json" ]]; then
    log "Check OK: $target_dir/model.safetensors.index.json"
  elif compgen -G "$target_dir/model-*.safetensors" >/dev/null; then
    log "Check OK: safetensors shard files exist in $target_dir"
  else
    log "Check MISSING: model.safetensors or model.safetensors.index.json or model-*.safetensors"
    missing=1
  fi

  return "$missing"
}

download_one_model() {
  local repo_id="$1"
  local target_dir
  target_dir="$(target_dir_for "$repo_id")"

  MODEL_STATUS["$repo_id"]="failed"
  MODEL_SOURCE["$repo_id"]="-"
  MODEL_PATH["$repo_id"]="$target_dir"

  log "============================================================"
  log "Start model: $repo_id"
  log "Default model root: $MODEL_DIR"
  log "Log file: $LOG_FILE"

  if download_hf_mirror "$repo_id" "$target_dir"; then
    log "HF-Mirror download command succeeded for $repo_id."
    if has_required_files "$target_dir"; then
      MODEL_STATUS["$repo_id"]="success"
      MODEL_SOURCE["$repo_id"]="HF-Mirror"
      log "SUCCESS: $repo_id downloaded and verified from HF-Mirror."
      return 0
    fi
    log "HF-Mirror command succeeded, but required file check failed. Will try ModelScope."
  else
    log "HF-Mirror download command failed for $repo_id. Will try ModelScope."
  fi

  if download_modelscope "$repo_id" "$target_dir"; then
    log "ModelScope download command succeeded for $repo_id."
    if has_required_files "$target_dir"; then
      MODEL_STATUS["$repo_id"]="success"
      MODEL_SOURCE["$repo_id"]="ModelScope"
      log "SUCCESS: $repo_id downloaded and verified from ModelScope."
      return 0
    fi
    log "ModelScope command succeeded, but required file check failed."
  else
    log "ModelScope download failed for $repo_id."
  fi

  MODEL_STATUS["$repo_id"]="failed"
  MODEL_SOURCE["$repo_id"]="-"
  log "FAILED: $repo_id was not fully downloaded or did not pass file checks."
  return 1
}

main() {
  local repo_id

  log "Download started."
  log "Model directory: $MODEL_DIR"
  log "Log directory: $LOG_DIR"
  log "Log file: $LOG_FILE"
  log "HF endpoint: $HF_ENDPOINT"
  log "Install ModelScope if missing: $INSTALL_MODELSCOPE"
  log "Pip index for optional ModelScope install: $PIP_INDEX_URL"
  log "Python: $(python3 --version 2>&1 || true)"
  log "hf: $(command -v hf || true)"
  log "huggingface-cli: $(command -v huggingface-cli || true)"
  log "modelscope: $(command -v modelscope || true)"

  for repo_id in "${MODELS[@]}"; do
    download_one_model "$repo_id" || true
  done

  log "============================================================"
  log "Summary"
  for repo_id in "${MODELS[@]}"; do
    log "$repo_id | status=${MODEL_STATUS[$repo_id]} | source=${MODEL_SOURCE[$repo_id]} | path=${MODEL_PATH[$repo_id]}"
  done
  log "Download finished. Full log: $LOG_FILE"
}

main "$@"
