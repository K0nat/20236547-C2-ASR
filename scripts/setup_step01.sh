#!/usr/bin/env bash
# STEP 01: 在 /root/siton-tmp 建立工作区并配置 multimodal 环境
# 用法: bash setup_step01.sh

set -euo pipefail

PROJECT_ROOT="/root/siton-tmp/speech_project"
CONDA_ENVS="/root/siton-tmp/conda_envs"
CONDA_PKGS="/root/siton-tmp/conda_pkgs"
CACHE_ROOT="/root/siton-tmp/cache"
ENV_NAME="multimodal"

echo "==> [1/7] 创建目录"
mkdir -p "$PROJECT_ROOT"/{docs,scripts,common_data/raw,common_data/processed}
mkdir -p "$PROJECT_ROOT/C2_ASR"/{code,outputs,logs}
mkdir -p "$CONDA_ENVS" "$CONDA_PKGS" "$CACHE_ROOT"/{huggingface,pip}

echo "==> [2/7] 配置 conda 使用 siton-tmp（不占用根分区 /）"
conda config --prepend envs_dirs "$CONDA_ENVS"
conda config --prepend pkgs_dirs "$CONDA_PKGS"

echo "==> [3/7] 写入 ~/.bashrc 环境变量"
grep -q 'SPEECH_PROJECT_ROOT' ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'EOF'

# speech_project · siton-tmp 工作区
export SPEECH_PROJECT_ROOT=/root/siton-tmp/speech_project
export HF_HOME=/root/siton-tmp/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PIP_CACHE_DIR=/root/siton-tmp/cache/pip
EOF
# shellcheck disable=SC1090
source ~/.bashrc

echo "==> [4/7] 创建 conda 环境: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "    环境 ${ENV_NAME} 已存在，跳过创建"
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> [5/7] 安装 PyTorch (CUDA 12.1) 与项目依赖"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r "$PROJECT_ROOT/requirements.txt"

echo "==> [6/7] 验证 Python / CUDA / 依赖"
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
"

echo "==> [7/7] 运行 verify_env.py"
cd "$PROJECT_ROOT/C2_ASR/code"
python verify_env.py

echo ""
echo "=== STEP 01 完成 ==="
echo "项目目录: $PROJECT_ROOT"
echo "激活环境: conda activate $ENV_NAME"
echo "下一步:   准备测试音频 -> common_data/raw/"
