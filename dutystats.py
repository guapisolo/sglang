#!/usr/bin/env python3
"""Clean overlap stats for an SGLang cuda-graph trace (robust to EP multi-stream).

Reports the numbers that matter for the v2-vs-v1 A/B:
  TRUE_DUTY  : GPU-active time / span, union across ALL kernel streams (true idle)
  STEP       : mean scheduler step period (between consecutive cudaGraphLaunch)
  BLK/STEP   : blocking cudaStreamSynchronize+Memcpy time on the launcher thread per step
  SYNC/STEP  : # cudaStreamSynchronize per step on launcher thread
Usage: dutystats.py <trace.json.gz>
"""

import gzip
import json
import statistics
import sys
from bisect import bisect_right
from collections import defaultdict

p = sys.argv[1]
d = json.load(gzip.open(p, "rt"))
ev = d["traceEvents"] if isinstance(d, dict) else d
X = [e for e in ev if e.get("ph") == "X"]
K = [e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
# union of all kernel streams
iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in K)
m = []
for s, en in iv:
    if m and s <= m[-1][1]:
        m[-1] = (m[-1][0], max(m[-1][1], en))
    else:
        m.append((s, en))
span = m[-1][1] - m[0][0]
busy = sum(en - s for s, en in m)
gaps = [m[i + 1][0] - m[i][1] for i in range(len(m) - 1)]
big = [g for g in gaps if g > 50]
# launcher thread = most cudaGraphLaunch
gl = defaultdict(list)
for e in X:
    if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", ""):
        gl[(e["pid"], e["tid"])].append(e["ts"])
lt = max(gl, key=lambda k: len(gl[k])) if gl else None
launches = sorted(gl[lt]) if lt else []
blk = [
    e
    for e in X
    if lt
    and (e["pid"], e["tid"]) == lt
    and e.get("cat") == "cuda_runtime"
    and ("Synchronize" in e["name"] or "Memcpy" in e["name"])
]
bts = [e["ts"] for e in sorted(blk, key=lambda e: e["ts"])]
blk_s = sorted(blk, key=lambda e: e["ts"])
per_blk, per_sync = [], []
for i in range(len(launches) - 1):
    lo, hi = launches[i], launches[i + 1]
    a, b = bisect_right(bts, lo), bisect_right(bts, hi)
    seg = blk_s[a:b]
    per_blk.append(sum(e.get("dur", 0) for e in seg))
    per_sync.append(sum(1 for e in seg if "Synchronize" in e["name"]))
nstep = len(launches)
print(f"=== {p.split('/')[-1]}")
print(f"  streams={len(set((e['pid'],e['tid']) for e in K))} cudaGraphLaunch={nstep}")
print(
    f"  TRUE_DUTY={100*busy/span:5.1f}%  idle={100*(span-busy)/span:4.1f}%  span={span/1000:.0f}ms active={busy/1000:.0f}ms"
)
if big:
    print(
        f"  GPU bubbles>50us: n={len(big)} median={statistics.median(big):.0f}us sum={sum(big)/1000:.0f}ms ({100*sum(big)/span:.0f}% of span)"
    )
if nstep > 1:
    print(f"  STEP period mean={span/(nstep-1)/1000:.2f}ms")
if per_blk:
    print(
        f"  BLK/STEP launcher-thread sync+memcpy: median={statistics.median(per_blk):.0f}us mean={sum(per_blk)/len(per_blk):.0f}us"
    )
if per_sync:
    print(
        f"  SYNC/STEP cudaStreamSynchronize count: median={statistics.median(per_sync):.1f} (intervals with >=1 sync: {sum(1 for c in per_sync if c>0)}/{len(per_sync)})"
    )
