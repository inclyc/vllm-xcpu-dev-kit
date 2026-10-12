#!/bin/bash
# Qwen3.5-0.8B compile; DP=1, PP=1, local MP, TP=1

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

# 在加载模板前设置独立配置项
export PD_MODE="NOT_MOE"

# 加载基础模板配置
source "$SCRIPT_DIR/user_env_template.sh"

# Use one local worker, matching the serial eager preset.
unset USER_VLLM_MP_RPC_WORKER_PER_NODE

# 覆盖必要配置
export USER_VLLM_EAGER_OR_NOT=""
export TORCH_XCPU_ENABLE_CHECK=0
export VLLM_OPTIONAL_ARGS="${VLLM_OPTIONAL_ARGS} --skip-mm-profiling"
export USER_VLLM_MODEL="Qwen/Qwen3.5-0.8B"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
unset VLLM_DISABLE_TQDM_AND_MONITOR

export VLLM_XCPU_GDN_DECODE_ONLY_COMPILE=1
_VLLM_OPTIONAL_ARGS+=" --reasoning-parser qwen3 --language-model-only"
export VLLM_OPTIONAL_ARGS="${_VLLM_OPTIONAL_ARGS}"

# 自动获取预设名称和目录
preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
if [ "$preset_dir" = "presets" ]; then
    echo "🚀 Preset: ${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
else
    echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
fi
