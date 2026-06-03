# Nemotron spec-v2 × overlap-scheduler 复现包

目标:验证 nemotron(super 先行)开 MTP(NEXTN) spec-v2 后,**overlap scheduler 是否真生效**——
即同事的问法:当前 step 的 `cudaGraphLaunch()` 应触发**下一**步的 kernel(CPU 领先 GPU 一步),
而不是每步被 GPU→CPU 同步抽干。用 torch profiler trace 证实。

## 整体思路

1. **必须开 cuda graph**。eager 下没有 `cudaGraphLaunch`,每步几百个零散 kernel launch,无法干净判读
   "launch 是否领先一步"。所以 `script.sh` 现在默认 `CUDA_GRAPH=1`。
2. **必须用 A/B 隔离**。dp-attention 每步本身就有 `all_gather` + `.cpu()/.item()` 同步(架构必需,与 spec 无关)。
   单看 v2 的绝对数字会把 dp-attention 的锅算到 spec 头上。所以跑三组、单变量对比:
   - `v2`    = spec v2 + overlap + cuda graph(被测对象)
   - `nospec`= 不开 spec + overlap + cuda graph(隔离 dp-attention 自身的 per-step sync)
   - `v1`    = spec + **关** overlap + cuda graph(串行参照,"被打破"长什么样)
3. **必须纯 decode 采样**。连续 batch 会把 chunked-prefill 混进 decode step → 噪声。
   用固定 batch(num-prompts == 并发)采,profiler 抓到的是干净的稳态 decode。
4. **指标**(`dutystats.py` / `analyze_graph.py`):
   - `TRUE_DUTY` = GPU 活跃时间 / 墙钟(对全部 stream 取并集;EP=8 有 ~94 条 stream,单看一条没意义)
   - 每步 `cudaStreamSynchronize` 次数 + 每步在 launcher 线程上的阻塞 ms
   - `cudaGraphLaunch` 发出时 GPU 是否仍在忙(CPU 领先率)
   判读:overlap 生效 ⇒ GPU duty 高、相邻 launch 间无把 GPU 抽干的长 sync;
        被打破 ⇒ 每个 launch 后紧跟长 sync、GPU 出现周期性 bubble。

## 文件清单(都在 repo 根目录,迁移时一起拷)

| 文件 | 作用 |
|---|---|
| `script.sh` | 启动 server(可选 gsm8k + bs1 profile)。所有实验开关在这里 |
| `profile_steady.sh` | 对已启动的 server 抓**稳态 decode** trace(防崩:先清残留 profiler、负载死则 abort、单会话) |
| `dutystats.py` | 出核心数字:TRUE_DUTY / 每步 sync 数 / 每步阻塞 ms / step 周期 |
| `analyze_graph.py` | cuda-graph 专用:CPU领先率[A] / launch间阻塞[B] / GPU bubble[C] |
| `analyze_trace.py` | eager trace 用(per-thread 阻塞分解) |

## script.sh 用法(env 开关)

```
MODEL_NAME=super|ultra     # 模型(默认 super)。或直接 MODEL=/abs/path 覆盖
SPEC=1|0                   # 开/关 MTP(NEXTN) spec(默认 1)
OVERLAP=1|0                # 1=spec v2(overlap+extra_buffer+radix); 0=spec v1(no overlap+no_buffer+no radix)
CUDA_GRAPH=1|0             # 默认 1(cuda graph 开)
PROFILE=0|1               # 1 => boot 后用 send_one 抓一个 bs1 decode trace
PROFILE_STEPS=12          # PROFILE=1 时抓多少步
MAX_RUNNING=16            # 同时也是 --cuda-graph-max-bs(graph 捕获到这个 bs)
N / PARALLEL              # gsm8k 题数 / 并发
MEMFRAC=0.6               # 共享节点上别人占了显存就调低
```

### 实验矩阵(逐条跑)

```bash
# v2(被测):boot + gsm8k,服务器留在 tmux 'nemo_srv'
MODEL_NAME=super SPEC=1 OVERLAP=1 CUDA_GRAPH=1 N=20 PARALLEL=16 MAX_RUNNING=16 bash script.sh

# nospec(隔离 dp-attention)
MODEL_NAME=super SPEC=0 OVERLAP=1 CUDA_GRAPH=1 N=20 PARALLEL=16 MAX_RUNNING=16 bash script.sh

# v1(串行参照)
MODEL_NAME=super SPEC=1 OVERLAP=0 CUDA_GRAPH=1 N=20 PARALLEL=16 MAX_RUNNING=16 bash script.sh
```

### 抓 trace

```bash
# bs1 最简单:boot 时直接带 PROFILE=1(用 send_one,单请求 decode)
MODEL_NAME=super PROFILE=1 PROFILE_STEPS=20 bash script.sh

# 任意 batch 的稳态 decode(server 已在跑):profile_steady.sh <tag> <num_steps> <并发>
#   并发=2  -> bs1/rank (dp2)     并发=16 -> bs8/rank     并发=32 -> bs16/rank
bash profile_steady.sh v2_bs1   40 2     # 你要的 bs1
bash profile_steady.sh v2_bs8   40 16

# 分析(trace 在打印的 TRACE_DIR 下,看 TP-0-DP-0)
python3 dutystats.py    <某个 ...-TP-0-DP-0-EP-0.trace.json.gz>
python3 analyze_graph.py <同上>
# 也可拖进 https://ui.perfetto.dev 看 'sglang::schedul' 线程的 cudaGraphLaunch / cudaStreamSynchronize
```

## 迁移到新机器的注意事项

1. **模型路径**:本机是 `/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` 和
   `/models/ea_nvidia_nemotron_3_ultra_550b_a55b_bf16_rl_050826_vv0.1`。新机器若不同,改 `script.sh` 里
   `MODEL_NAME` 分支的路径,或运行时 `MODEL=/新路径 ...`。
2. **`/cluster-storage` 可能不存在**:`script.sh` 的 `LOG`/`PROFILE_DIR` 和 `profile_steady.sh` 的 `OUT`
   默认写 `/cluster-storage/bench_runs`。新机器没有就 `LOG=/tmp/... PROFILE_DIR=/tmp/...` 覆盖,
   并把 `profile_steady.sh` 里的 `OUT=/cluster-storage/...` 改成本地盘。
3. **显存**:独占机器可把 `MEMFRAC` 调回 0.85+;若仍有别人占卡就保持 0.6。
4. **拓扑**:super 120B 单节点 8 卡够;**ultra 550B 权重 ~137GB/卡,8×B200 单节点只够 eager,
   且要独占**(被别人占 ~55GB 时放不下)。

## 当前进展(本机已得到的)

- ✅ super tp8 ep8 dp2 + spec v2(NEXTN) + cuda graph 跑通,gsm8k 0.90~0.95,accept len ~1.78。
  确认走的是 `EAGLEWorkerV2`(spec v2,overlap 未关)。
- ✅ **v2 干净 trace(固定 batch 8/rank,纯 decode)**:`TRUE_DUTY=70.7%`(idle 29.3%),step 33.7ms,
  GPU实算/step ≈23ms,**launcher 线程每步阻塞 ~14ms、14 个 `cudaStreamSynchronize`(38/38 步都有)**。
  → 即 `step ≈ GPU + sync`,sync 大部分**没被 GPU 计算盖住**,CPU 没干净领先一步,GPU 空 29%。
  trace: `/cluster-storage/bench_runs/trace_v2_clean_080133/.../...-TP-0-DP-0-EP-0.trace.json.gz`
- ⏳ **未完成**:nospec / v1 两个对照(本机被抢卡,nospec/ v1 启动失败)。迁移后补上即可定论:
  - nospec 若也 ~29% idle / 14 syncs ⇒ idle 是 dp-attention 的 all_gather,**spec v2 没额外破坏 overlap**;
  - nospec 若干净很多 ⇒ 是 spec 的锅。

## 坑(都已解决,且都不是 spec v2 的 bug)

- 服务器崩溃:我采集编排失误——挂了个没结束的 profiler 又叠了第二个 → scheduler OOM。已加固 `profile_steady.sh`。
- v1 启动失败:`extra_buffer` 与 `--disable-radix-cache` 冲突。已改为 overlap→`extra_buffer`+radix、
  no-overlap→`no_buffer`+no-radix。
