#!/usr/bin/env python3
"""Plan SRPT + recompute-front for G simultaneous restores, from this machine's measurements.

Restore i (queue position i) recomputes history[0:k_i) on its own GPU while it waits, then
loads [k_i, N) from the array. With W_i = finish time of restore i-1 (W_0 = 0):
    F_i = max(W_i, t(k_i)) + (N - k_i) * s_tok / B,   k_i = argmin F_i over multiples of 128.
t(n): vLLM prefill time measured in stage 30 (linear interpolation, t(0) = 0).
B: single-restore bandwidth measured in stage 20 (G=1 median, bytes / restore_s).

Writes schedule.json: per geometry {"G", "B", "t_source", "k": [...], "delay_s": [...] (= t(k_i),
the emulated recompute time), "pred": {fair, srpt, srpt_front: mean/max completion}}.

  python3 front_schedule.py --concurrent CSV --prefill CSV|none --G 8 --out schedule.json
"""
import argparse
import csv
import json
import statistics

N = 32768
TOK = {"8b": 32 * 8 * 128 * 2 * 2, "70b": 80 * 8 * 128 * 2 * 2}  # KV bytes per token (BF16)
# Fallback t(n) for 8B measured on H200 (vLLM 0.10.1.1, FP8 weights, BF16 KV)
H200_8B = {0: 0.0, 4096: 0.067, 8192: 0.150, 16384: 0.370, 32768: 0.996}
H200_70B_OVER_8B = 7.01 / 0.996  # 70B / 8B prefill ratio at 32k on H200

ap = argparse.ArgumentParser()
ap.add_argument("--concurrent", required=True)
ap.add_argument("--prefill", required=True)
ap.add_argument("--G", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def interp(pts, n):
    xs = sorted(pts)
    if n <= xs[0]:
        return pts[xs[0]]
    for x0, x1 in zip(xs, xs[1:]):
        if n <= x1:
            return pts[x0] + (pts[x1] - pts[x0]) * (n - x0) / (x1 - x0)
    return pts[xs[-1]] * n / xs[-1]


meas = {}
if a.prefill != "none":
    try:
        rows = [r for r in csv.DictReader(open(a.prefill)) if r["phase"] == "run"]
        for tag in {r["model"] for r in rows}:
            pts = {0: 0.0}
            for n in {int(r["tokens"]) for r in rows if r["model"] == tag}:
                pts[n] = statistics.median(float(r["wall_s"]) for r in rows
                                           if r["model"] == tag and int(r["tokens"]) == n)
            meas[tag] = pts
    except FileNotFoundError:
        pass

conc = list(csv.DictReader(open(a.concurrent)))
out = {}
for geo in ("8b", "70b"):
    single = [int(r["bytes"]) / float(r["restore_s"]) for r in conc
              if r["exp"] == f"concurrent_{geo}" and r["G"] == "1" and r["restore_s"]]
    if not single:
        continue
    B = statistics.median(single)
    if geo in meas:
        t, src = meas[geo], "measured"
    elif geo == "70b" and "8b" in meas:
        t = {n: v * H200_70B_OVER_8B for n, v in meas["8b"].items()}
        src = "ASSUMED: this machine's 8B t(n) x 7.04 (H200 70B/8B ratio at 32k)"
    elif geo == "8b":
        t, src = H200_8B, "FALLBACK: H200 8B t(n), not measured here"
    else:
        t = {n: v * H200_70B_OVER_8B for n, v in H200_8B.items()}
        src = "FALLBACK: H200 8B t(n) x 7.04, not measured here"
    s = TOK[geo]
    S = N * s
    G = a.G
    W, ks, fin = 0.0, [], []
    for _ in range(G):
        best = min(((max(W, interp(t, k)) + (N - k) * s / B, k) for k in range(0, N, 128)))
        fin.append(best[0])
        ks.append(best[1])
        W = best[0]
    srpt = [(i + 1) * S / B for i in range(G)]
    out[geo] = dict(G=G, B_bytes_per_s=B, t_source=src, t_points=t, k=ks,
                    delay_s=[interp(t, k) for k in ks],
                    pred=dict(fair=dict(mean=G * S / B, max=G * S / B),
                              srpt=dict(mean=sum(srpt) / G, max=srpt[-1]),
                              srpt_front=dict(mean=sum(fin) / G, max=max(fin)),
                              recompute_full=interp(t, N)))
json.dump(out, open(a.out, "w"), indent=1)
for geo, v in out.items():
    p = v["pred"]
    print(f"{geo}: B={v['B_bytes_per_s']/1e9:.1f} GB/s t:{v['t_source']} k={v['k']}")
    print(f"   predicted mean: fair {p['fair']['mean']:.3f}  srpt {p['srpt']['mean']:.3f}  "
          f"srpt+front {p['srpt_front']['mean']:.3f}  (recompute {p['recompute_full']:.3f})")
