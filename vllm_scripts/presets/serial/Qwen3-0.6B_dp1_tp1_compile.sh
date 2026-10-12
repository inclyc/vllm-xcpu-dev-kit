#!/bin/bash
# Preset: Qwen3-0.6B
# Configuration: DP=1, TP=1, PP=1, Inductor C++ wrapper

SCRIPT_DIR="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_DIR%/presets/*}"

# 在加载模板前设置独立配置项
export PD_MODE="NOT_MOE"
export USER_VLLM_MAX_NUM_BATCHED_TOKENS="${USER_VLLM_MAX_NUM_BATCHED_TOKENS:-32}"
export VLLM_TEST_MAX_WAIT="${VLLM_TEST_MAX_WAIT:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"

# 加载基础模板配置
source "$SCRIPT_DIR/user_env_template.sh"

# 覆盖必要配置
unset USER_VLLM_EAGER_OR_NOT
export TORCH_XCPU_ENABLE_CHECK=0
export USER_VLLM_MODEL="Qwen/Qwen3-0.6B"
export USER_VLLM_DATA_PARALLEL_SIZE=1
export USER_VLLM_TP_SIZE=1
export USER_VLLM_PP_SIZE=1
export USER_VLLM_MPC_SIZE=$((USER_VLLM_TP_SIZE * USER_VLLM_PP_SIZE))
unset USER_VLLM_MP_RPC_WORKER_PER_NODE
unset VLLM_DISABLE_TQDM_AND_MONITOR
export TORCHINDUCTOR_CPP_WRAPPER=1


# 自动获取预设名称和目录
preset_name=$(basename "${BASH_SOURCE[0]}" .sh)
preset_dir=$(basename "$(dirname "${BASH_SOURCE[0]}")")
if [ "$preset_dir" = "presets" ]; then
    echo "🚀 Preset: ${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
else
    echo "🚀 Preset: ${preset_dir}/${preset_name} | DP=${USER_VLLM_DATA_PARALLEL_SIZE}, TP=${USER_VLLM_TP_SIZE}, PP=${USER_VLLM_PP_SIZE}"
fi
