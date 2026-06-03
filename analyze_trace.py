#!/usr/bin/env python3
"""Analyze an SGLang torch-profiler Chrome trace for overlap-scheduler behavior.

Key questions:
  1. Is the GPU compute stream kept busy (high duty cycle, small inter-step bubbles)?
  2. Which CPU thread carries blocking syncs (cudaStreamSynchronize / blocking
     cudaMemcpy)?  The scheduler/launch thread blocking => overlap broken.
     A dedicated d2h-prepare thread blocking => fine (FutureMap design).
  3. Decode-step segmentation: per-step GPU time vs inter-step idle bubble.

Usage: python analyze_trace.py <trace.json.gz> [step_boundary_gap_us=80]
"""

import gzip
import json
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
STEP_GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0

op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    data = json.load(f)
events = data["traceEvents"] if isinstance(data, dict) else data

thread_name = {}
for e in events:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        thread_name[(e["pid"], e["tid"])] = e["args"].get("name", "")

X = [e for e in events if e.get("ph") == "X"]
kernels = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
by_stream = defaultdict(list)
for e in kernels:
    by_stream[(e["pid"], e["tid"])].append(e)

print(f"=== {path.split('/')[-1]}")
print(f"events={len(events)} kernels={len(kernels)} kernel-streams={len(by_stream)}")


def merge(evs):
    ivs = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in evs)
    m = []
    for s, en in ivs:
        if m and s <= m[-1][1]:
            m[-1] = (m[-1][0], max(m[-1][1], en))
        else:
            m.append((s, en))
    return m


main = max(by_stream, key=lambda k: sum(e.get("dur", 0) for e in by_stream[k]))
evs = by_stream[main]
busy = merge(evs)
span = busy[-1][1] - busy[0][0]
busy_t = sum(en - s for s, en in busy)
gaps = [busy[i + 1][0] - busy[i][1] for i in range(len(busy) - 1)]
step_bnd = [g for g in gaps if g > STEP_GAP]  # inter-step bubbles
print(f"\n--- GPU compute stream {main} '{thread_name.get(main,'?')}' ---")
print(
    f"  span={span/1000:.1f}ms busy={busy_t/1000:.1f}ms  DUTY={100*busy_t/span:.1f}%  idle={100*(span-busy_t)/span:.1f}%"
)
print(
    f"  step-boundary gaps (>{STEP_GAP}us): n={len(step_bnd)}"
    + (
        f"  median={statistics.median(step_bnd):.0f}us mean={sum(step_bnd)/len(step_bnd):.0f}us"
        f" sum={sum(step_bnd)/1000:.1f}ms"
        if step_bnd
        else ""
    )
)
if len(step_bnd) >= 3:
    # approximate per-step period = span / num_steps
    nsteps = len(step_bnd) + 1
    print(
        f"  ~{nsteps} steps  -> mean step period={span/nsteps/1000:.2f}ms"
        f"  bubble/period={100*statistics.median(step_bnd)/(span/nsteps):.1f}%"
    )

# second-busiest stream (often the forward/draft stream) for context
if len(by_stream) > 1:
    others = sorted(
        by_stream, key=lambda k: -sum(e.get("dur", 0) for e in by_stream[k])
    )
    for k in others[1:3]:
        b = merge(by_stream[k])
        bt = sum(en - s for s, en in b)
        print(
            f"  other stream {k} '{thread_name.get(k,'?')}': kernels={len(by_stream[k])} busy={bt/1000:.1f}ms"
        )

# ---- per-CPU-thread cuda_runtime blocking breakdown ----
rt = [e for e in X if e.get("cat") == "cuda_runtime"]
per = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))  # tid -> name -> [cnt, dur]
for e in rt:
    k = (e["pid"], e["tid"])
    n = e.get("name", "")
    per[k][n][0] += 1
    per[k][n][1] += e.get("dur", 0)
print(f"\n--- per CPU thread: blocking cuda_runtime time (Sync + Memcpy) ---")


def block_time(d):
    return sum(v[1] for n, v in d.items() if "Synchronize" in n or "Memcpy" in n)


for k in sorted(per, key=lambda k: -block_time(per[k])):
    d = per[k]
    launches = sum(v[0] for n, v in d.items() if "Launch" in n)
    blk = block_time(d)
    detail = "  ".join(
        f"{n.replace('cuda',''):14s}n={v[0]} {v[1]/1000:.0f}ms"
        for n, v in sorted(d.items(), key=lambda x: -x[1][1])
        if ("Synchronize" in n or "Memcpy" in n) and v[1] > 1000
    )
    print(
        f"  tid {k} '{thread_name.get(k,'?')[:34]}': launches={launches:6d}  BLOCKING={blk/1000:6.0f}ms | {detail}"
    )
