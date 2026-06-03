#!/usr/bin/env python3
"""Decisive overlap test for cuda-graph traces.

Answers your colleague's question directly:
  "current step's cudaGraphLaunch() should trigger the kernels for the NEXT step,
   not the current step."

Method:
  - launcher thread = thread with the most cudaGraphLaunch calls
  - GPU compute stream = busiest kernel stream; merge kernels into busy intervals
  - Metric A (CPU ahead?): fraction of cudaGraphLaunch calls issued while the GPU
      compute stream is STILL BUSY with earlier work. ~1.0 => CPU runs ahead of GPU
      => the launch is for a future step => overlap works.
  - Metric B (serialization tell): blocking cuda_runtime (Synchronize / blocking
      Memcpy) on the launcher thread BETWEEN consecutive graph launches. ~0 => no
      per-step CPU->GPU sync on the critical path. Large => spec serializes (v1).
  - Metric C (GPU bubble): inter-burst idle gap on the GPU compute stream.

Usage: python analyze_graph.py <trace.json.gz> [burst_gap_us=50]
"""

import gzip
import json
import statistics
import sys
from bisect import bisect_right
from collections import defaultdict

path = sys.argv[1]
BURST_GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data
tname = {
    (e["pid"], e["tid"]): e["args"].get("name", "")
    for e in events
    if e.get("ph") == "M" and e.get("name") == "thread_name"
}
X = [e for e in events if e.get("ph") == "X"]

# launcher thread = most cudaGraphLaunch
gl_by_t = defaultdict(list)
for e in X:
    if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", ""):
        gl_by_t[(e["pid"], e["tid"])].append(e["ts"])
if not gl_by_t:
    print("NO cudaGraphLaunch events -> trace is eager, use analyze_trace.py")
    sys.exit(0)
ltid = max(gl_by_t, key=lambda k: len(gl_by_t[k]))
launches = sorted(gl_by_t[ltid])
print(f"=== {path.split('/')[-1]}")
print(
    f"launcher thread {ltid} '{tname.get(ltid,'?')}'  cudaGraphLaunch={len(launches)}"
)

# GPU compute stream = busiest
kern = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
by_s = defaultdict(list)
for e in kern:
    by_s[(e["pid"], e["tid"])].append(e)
gpu = max(by_s, key=lambda k: sum(e.get("dur", 0) for e in by_s[k]))
ivs = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in by_s[gpu])
busy = []
for s, en in ivs:
    if busy and s <= busy[-1][1]:
        busy[-1] = (busy[-1][0], max(busy[-1][1], en))
    else:
        busy.append((s, en))
span = busy[-1][1] - busy[0][0]
busy_t = sum(en - s for s, en in busy)
starts = [b[0] for b in busy]


# Metric A: fraction of launches issued while GPU busy
def gpu_busy_at(t):
    i = bisect_right(starts, t) - 1
    return 0 <= i < len(busy) and busy[i][0] <= t < busy[i][1]


ahead = sum(1 for t in launches if gpu_busy_at(t))
print(
    f"\n[A] CPU-ahead: {ahead}/{len(launches)} = {100*ahead/len(launches):.1f}% of cudaGraphLaunch "
    f"issued while GPU still busy with earlier work"
)
print(f"    => high % means the launch feeds a FUTURE step (overlap working)")

# Metric C: GPU duty + inter-burst bubble
bursts = []
for s, en in busy:
    if bursts and s - bursts[-1][1] < BURST_GAP:
        bursts[-1] = (bursts[-1][0], en)
    else:
        bursts.append((s, en))
gaps = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
print(
    f"\n[C] GPU stream {gpu} '{tname.get(gpu,'?')}': span={span/1000:.1f}ms busy={busy_t/1000:.1f}ms "
    f"DUTY={100*busy_t/span:.1f}% idle={100*(span-busy_t)/span:.1f}%"
)
print(
    f"    graph bursts={len(bursts)}  inter-burst bubble: median={statistics.median(gaps):.0f}us "
    f"mean={sum(gaps)/len(gaps):.0f}us max={max(gaps):.0f}us  (>{BURST_GAP}us defines a burst boundary)"
    if gaps
    else ""
)

# Metric B: blocking runtime between consecutive launches on launcher thread
blk = [
    e
    for e in X
    if (e["pid"], e["tid"]) == ltid
    and e.get("cat") == "cuda_runtime"
    and ("Synchronize" in e.get("name", "") or "Memcpy" in e.get("name", ""))
]
blk.sort(key=lambda e: e["ts"])
bts = [e["ts"] for e in blk]
per_step_blk, sync_cnt = [], []
for i in range(len(launches) - 1):
    lo, hi = launches[i], launches[i + 1]
    a = bisect_right(bts, lo)
    b = bisect_right(bts, hi)
    seg = blk[a:b]
    per_step_blk.append(sum(e.get("dur", 0) for e in seg))
    sync_cnt.append(sum(1 for e in seg if "Synchronize" in e["name"]))
if per_step_blk:
    print(
        f"\n[B] blocking sync/memcpy on launcher thread BETWEEN consecutive cudaGraphLaunch:"
    )
    print(
        f"    per-interval blocking: median={statistics.median(per_step_blk):.0f}us "
        f"mean={sum(per_step_blk)/len(per_step_blk):.0f}us max={max(per_step_blk):.0f}us"
    )
    print(
        f"    intervals with a Synchronize: {sum(1 for c in sync_cnt if c>0)}/{len(sync_cnt)}"
    )
    print(
        f"    => ~0 blocking between launches means no per-step CPU<->GPU serialization"
    )

# breakdown of launcher-thread blocking calls
agg = defaultdict(lambda: [0, 0.0])
for e in X:
    if (e["pid"], e["tid"]) == ltid and e.get("cat") == "cuda_runtime":
        n = e["name"]
        if "Synchronize" in n or "Memcpy" in n or "GraphLaunch" in n:
            agg[n][0] += 1
            agg[n][1] += e.get("dur", 0)
print(f"\n    launcher-thread cuda_runtime totals:")
for n, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print(f"      {n:26s} n={c:6d} total={d/1000:8.1f}ms mean={d/c:6.1f}us")
