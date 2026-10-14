# 完整 GLM5.2 真实权重验证

运行日期：2026-10-14。宿主机为单台 x86 Xeon Gold 6248，容器 CPU 配额
32，内存上限由用户提高到 900 GiB。四个 MPI 进程组成 DP2/TP2/EP4，
每组 TP2 使用独立缓存空间。**这是单机四进程验证，不是双节点验收。**

## 配置

- 原始 78 层模型、hidden size 6144、256 experts、top-8，无尺寸覆盖。
- 真实 safetensors 权重；MXFP4 使用现有 W4A16 路径，保留压缩权重，
  激活为 BF16，与仓库现有完整模型 preset 一致。
- eager V1、同步调度、`mpi_alltoallv_v6`、FP8 KV、prefix caching 开启。
- 每个 rank 的逻辑 KV 预算 128 MiB。共享模式每组 TP2 配置 256 MiB arena，
  位于 `/dev/shm`，实际 KV backing 以 worker 返回值为准。
- 每个 DP 一个不同的中/英文 chat 请求，greedy 生成 6 tokens，重复两轮。
  请求模板后分别为 24/21 tokens，均少于 64-token cache block，因此此轮
  不覆盖 prefix cache 命中、长上下文或 index top-k=2048 的跨界路径。
- 6 tokens 仅验证生成路径和数值一致性，不能作为完整回答质量评价。

测试程序：[model-smoke.py](model-smoke.py)；独立比较程序：
[compare-full.py](compare-full.py)。原始日志和输出在
`/workspace/logs/shared-kv-full-validation/`。

## 复现

从 `/workspace` 运行，使用新的输出目录和 arena 路径。先运行 private，
进程全部退出后再运行 shared，避免同时驻留两份完整模型。

```bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
unset TORCH_MCPU_SHARED_BASE
run_dir=$(mktemp -d /workspace/logs/shared-kv-full-recheck.XXXXXX)
model=/model-cache/huggingface/hub/models--amd--GLM-5.2-MXFP4/snapshots/386bd0e4ec821f7b07975701cec3c3b953a5576a
for mode in private shared; do
  export VLLM_XCPU_SHARED_KV_CACHE=0
  if [ "$mode" = shared ]; then export VLLM_XCPU_SHARED_KV_CACHE=1; fi
  export TORCH_MCPU_SHARED_SIZE=268435456
  export TORCH_MCPU_SHARED_SIM_FILE="/dev/shm/$(basename "$run_dir")-mxfp4"
  timeout -k 30 14400 /workspace/.venv/bin/python \
    docs/shared-kv-cache/model-smoke.py --model "$model" \
    --weights MXFP4-W4A16 --dp-size 2 --expert-parallel --timeout 14000 \
    --output "$run_dir/$mode-mxfp4.json" > "$run_dir/$mode-mxfp4.log" 2>&1
done
/workspace/.venv/bin/python docs/shared-kv-cache/compare-full.py \
  "$run_dir" --weights mxfp4
```

比较程序检查两组 DP 的完整输出（token、文本、top logprob）完全一致、
logprob 有限、缓存容量一致、共享文件关系、恰有一个写入者和缓存内容哈希。
共享 arena 文件只能在所有 worker 退出后删除。

## 运行记录

首次 MXFP4 私有模式已成功加载和预热，但测试脚本未显式设置 tokenizer
`return_dict=False`，将新版本返回的字典误当成 token 列表，提交请求失败。
修正后重跑的私有基线成功。首次异常退出伴随 MPI 清理错误，保留在
`private-mxfp4-tokenizer-error.log`；不计为通过。

私有基线实测约 439 GiB 常驻内存，cgroup 含文件缓存峰值约 708 GiB，
未触发内存限额。四个 rank 每个约 108.4 GiB 模型内存。

MXFP4 私有/共享两种模式均正常退出。两组 DP 的 token、文本、top logprob
逐项完全一致，所有 logprob 有限。两组 TP 各自使用同一 inode 和相同缓存
SHA256；跨 DP inode 不同，每组恰有一个写入者。

每个 rank 逻辑容量为 38 blocks / 2,432 tokens，KV backing 为 131,182,080
字节。四进程私有总量 524,728,320 字节；共享后为两份，总量 262,364,160
字节，减少 50%。这是短请求测试配置下的缓存数据量，不是整机 RSS 减半。

FP8 私有基线也已成功加载、预热、完成两轮生成并正常退出。常驻内存
峰值约 743 GiB，cgroup 含文件缓存峰值接近 900 GiB，未发生 OOM。
FP8 共享模式也正常完成，两组 DP 的 token、文本、top logprob 与私有模式
完全一致。容量、单写入者、共享 inode、缓存 SHA256、跨 DP 隔离检查均通过。
FP8 的 KV backing 字节数与 MXFP4 相同，私有 524,728,320 字节，
共享 262,364,160 字节，减少 50%。

两种真实权重的结果汇总见 [full-validation-results.json](full-validation-results.json)。
两种模式退出均有 Python resource_tracker 清理共享内存的警告；成功轮次退出码
均为 0。测试没有覆盖双物理节点、长序列、压满容量、compile/异步调度或吞吐性能。
真实硬件上的共享空间与内存可见性仍需另行验收。

重跑 FP8 时，将上述命令的 `model` 改为
`/model-cache/modelscope/hub/models/ZhipuAI/GLM-5.2-FP8`，`--weights` 改为
`FP8`，输出文件前缀和 arena 路径中的 `mxfp4` 改为 `fp8`，比较程序使用
`--weights fp8`。
