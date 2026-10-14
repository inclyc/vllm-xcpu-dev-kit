# Shared KV cache validation — 2026-10-14

后续完整真实权重验证记录见 [full-validation.md](full-validation.md)。本页保留
最初缩小模型的测试范围与结果。

第一版已接入现有 `xmalloc`。GLM5.2 的 MLA 主缓存和 indexer 缓存在每组
TP2 内共用物理存储；只有 TP rank 0 写入。页表仍由现有 scheduler/worker
维护，各 worker 保留自己的表，共享的是表所指向的缓存数据。

实现说明见 [插件文档](../../vllm-xcpu-plugin/docs/shared_kv_cache.md)，
分配接口见 [xmalloc 文档](../../torch_mcpu/docs/shared_xmalloc.md)。

## 验证结果

所有模型测试均为 **x86 上缩小维度的七层 GLM5.2、dummy 权重、eager V1**。
FP8 和 MXFP4-W4A16 指权重量化格式；两者均使用 FP8 MLA KV cache。
这验证了实际模型执行路径，不代表完整预训练模型的质量或性能验证。

| 权重格式 | 并行配置 | 共享前后 token / logprob | 原缓存数据总字节 | 共享后缓存数据总字节 |
| --- | --- | --- | ---: | ---: |
| FP8 | DP1 / TP2 | 完全一致 | 16,384,000 | 8,192,000 |
| MXFP4-W4A16 | DP1 / TP2 | 完全一致 | 16,384,000 | 8,192,000 |
| FP8 | DP2 / TP2 / EP4 | 两个 DP 均完全一致 | 32,768,000 | 16,384,000 |
| MXFP4-W4A16 | DP2 / TP2 / EP4 | 两个 DP 均完全一致 | 32,768,000 | 16,384,000 |

每个 TP rank 的逻辑容量相同：25 blocks、1,600 tokens。worker RPC 确认
同组两个 rank 的文件 inode 和缓存内容 SHA256 相同，且恰有一个写入者；
两个 DP 使用不同文件。启动时另有 nonce 写入/读取探针检查真实共享可见性。
上述字节数是缓存 backing 去重后的数据大小，不是进程 RSS、整机节省量或
配置的 arena 大小；本测试为每组配置 16 MiB arena，实际使用约 7.81 MiB。

- 13 个插件测试通过：两组 TP、普通/packed 分配、真实 MLA/indexer 写入、
  故意不同的 follower 输入、slot 覆写复用、Q 结果、状态更新写权限和既有路由回归。
- 6 个 xmalloc 测试通过：对齐/容量/生命周期、跨进程共享与隔离、非法配置、
  文件大小不匹配、固定基址保护及预映射 C++ 分支。
- 本次改动的插件 Python 文件通过 Ruff 检查，Git diff 空白检查通过。

生成测试使用两个请求、分块 prefill、6 步 greedy decode，并重复请求覆盖前缀复用。
DP2 测试给两个 DP 不同输入；使用单机四进程 MPI external launcher，
`mpi_alltoallv_v6`、8 experts/top-8。尚未验证真实双节点部署。

结果汇总在 [validation-results.json](validation-results.json)。原始输出、worker
报告与日志位于 `/workspace/logs/shared-kv-validation/`。DP2 FP8 成功日志是
`{shared,private}-dp2-fp8-mpi2.log`，MXFP4 是 `{shared,private}-dp2-mxfp4.log`。
MPI 退出时两种模式均有 Python resource_tracker 清理共享内存的警告；进程正常退出。

## 复现

在 `/workspace` 执行。以下命令会完整重跑八个模型实例；使用新目录和新 arena 路径。

```bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export VLLM_USE_V2_MODEL_RUNNER=0
unset TORCH_MCPU_SHARED_BASE
run_dir=$(mktemp -d /workspace/logs/shared-kv-recheck.XXXXXX)

for weights in FP8 MXFP4-W4A16; do
  label=fp8
  if [ "$weights" = MXFP4-W4A16 ]; then label=mxfp4; fi
  for dp in 1 2; do
    name=$label
    extra=()
    if [ "$dp" = 2 ]; then
      name=dp2-$label
      extra=(--dp-size 2 --expert-parallel)
    fi
    for mode in private shared; do
      export VLLM_XCPU_SHARED_KV_CACHE=0
      if [ "$mode" = shared ]; then export VLLM_XCPU_SHARED_KV_CACHE=1; fi
      export TORCH_MCPU_SHARED_SIZE=16777216
      export TORCH_MCPU_SHARED_SIM_FILE="$run_dir/arena-$name"
      timeout -k 15 300 /workspace/.venv/bin/python \
        docs/shared-kv-cache/model-smoke.py --weights "$weights" \
        "${extra[@]}" --output "$run_dir/$mode-$name.json" \
        > "$run_dir/$mode-$name.log" 2>&1
    done
  done
done
/workspace/.venv/bin/python docs/shared-kv-cache/compare-smoke.py "$run_dir"
```

测试和比较脚本会断言输出一致、容量一致、写入者数量和 TP/DP 文件关系。
脚本只在本地可信进程间使用 callback RPC，因此在测试进程中开启了
`VLLM_ALLOW_INSECURE_SERIALIZATION`；功能本身不依赖这个选项。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/.venv/bin/python -m pytest \
  vllm-xcpu-plugin/tests/plugin_tests/test_shared_kv_cache.py \
  vllm-xcpu-plugin/tests/layers/test_mla_rope_kvcache_routing.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/.venv/bin/python -m pytest \
  torch_mcpu/tests/test_memory.py -k TestSharedXmalloc -q
```

## 当前边界

同步屏障优先保证正确性，未做延迟/吞吐优化。只启用 eager、同步调度的 V1
GLM sparse MLA 路径；不支持 compile、异步调度、推测解码或 AF-EP。
真实硬件需要提供预映射区域，并确认设备同步能保证跨进程可见性；非一致性
内存还需要硬件发布/失效操作。完整尺寸模型与真实硬件尚未验证。

本次使用现有 MPI 路径完成 EP 验证。尝试的非 MPI `all_to_all_single` 和
`torch_all_to_all_single` 在私有缓存基线中也分别遇到了 Modular Prepare/Finalize
不匹配和缺少 `_count_expert_num_tokens` fake Triton kernel；未将这些独立问题纳入修改。
