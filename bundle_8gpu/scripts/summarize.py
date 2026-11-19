#!/usr/bin/env python3
"""Collect every stage's results into OUT_DIR/summary.json and OUT_DIR/summary.md.
Safe to run at any point; missing stages are reported as missing.

  python3 summarize.py OUT_DIR
"""
import csv
import glob
import json
import os
import statistics
import sys

O = sys.argv[1]
S = {"stages": {}}


def med(v):
    return statistics.median(v) if v else None


def rows(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


for p in sorted(glob.glob(f"{O}/*/status.json")):
    s = json.load(open(p))
    S["stages"][s["stage"]] = {k: s[k] for k in ("status", "note")}

# machine
gp = f"{O}/00_preflight/gpus.csv"
if os.path.exists(gp):
    g = list(csv.reader(open(gp)))
    S["gpus"] = [", ".join(x.strip() for x in r) for r in g[1:]]
sp = f"{O}/00_preflight/summary.txt"
if os.path.exists(sp):
    S["preflight"] = dict(l.split(": ", 1) for l in open(sp).read().splitlines() if ": " in l)

# fio
F = {}
for bs in ("1M", "512K", "256K"):
    v = [json.load(open(p))["jobs"][0]["read"]["bw_bytes"] / 1e9
         for p in glob.glob(f"{O}/10_fio/read_{bs}_r*.json")]
    if v:
        F[f"read_{bs}_GBps_median"] = med(v)
p = f"{O}/10_fio/read_under_write.json"
if os.path.exists(p):
    j = json.load(open(p))["jobs"]
    rd, wr = j[0], j[1] if len(j) > 1 else None
    F["mixed_read_GBps"] = rd["read"]["bw_bytes"] / 1e9
    F["mixed_read_p99_ms"] = rd["read"]["clat_ns"]["percentile"].get("99.000000", 0) / 1e6
    if wr:
        F["mixed_write_GBps"] = wr["write"]["bw_bytes"] / 1e9
S["fio"] = F


def group(rs, key):
    out = {}
    for r in rs:
        out.setdefault(key(r), []).append(r)
    return out


def completion_stats(rs, from_arrival=False):
    """Per repetition: mean / max completion over processes; then median over repetitions."""
    per_rep = group(rs, lambda r: r["rep"])
    means, maxs, aggbw = [], [], []
    for rep, rr in per_rep.items():
        # exit is empty while the process was still running (multi-repetition launch)
        if any(not r["end_rel_s"] or r["exit"] not in ("", "0") or r["mismatches"] != "0"
               for r in rr):
            continue
        c = [float(r["end_rel_s"]) - (float(r["delay_s"]) if from_arrival else 0) for r in rr]
        means.append(sum(c) / len(c))
        maxs.append(max(c))
        aggbw.append(sum(int(r["bytes"]) for r in rr) / max(float(r["end_rel_s"]) for r in rr) / 1e9)
    return dict(reps=len(means), mean_s=med(means), max_s=med(maxs),
                mean_s_all_reps=means, agg_GBps=med(aggbw))


def table(csvpath, from_arrival=False):
    res = {}
    for (exp, G, pol), rr in group(rows(csvpath), lambda r: (r["exp"], r["G"], r["policy"])).items():
        res.setdefault(exp, {}).setdefault(f"G{G}", {})[pol] = completion_stats(rr, from_arrival)
    for exp in res.values():
        for gk, d in exp.items():
            if "fair" in d and "srpt" in d and d["srpt"]["mean_s"]:
                d["srpt_vs_fair_mean_speedup"] = d["fair"]["mean_s"] / d["srpt"]["mean_s"]
    return res


S["concurrent"] = table(f"{O}/20_concurrent/concurrent.csv")
S["writeback"] = table(f"{O}/21_writeback/writeback.csv")
S["front"] = table(f"{O}/31_front_emul/front.csv")
S["arrivals"] = table(f"{O}/32_arrivals/arrivals.csv", from_arrival=True)
sch = f"{O}/31_front_emul/schedule.json"
if os.path.exists(sch):
    S["front_schedule"] = {g: {k: v for k, v in d.items() if k != "t_points"}
                           for g, d in json.load(open(sch)).items()}

# t(n)
T = {}
for r in rows(f"{O}/30_prefill/prefill.csv"):
    if r["phase"] == "run":
        T.setdefault(r["model"], {}).setdefault(int(r["tokens"]), []).append(float(r["wall_s"]))
S["prefill_s"] = {m: {n: med(v) for n, v in sorted(d.items())} for m, d in T.items()}

# e2e
p = f"{O}/40_e2e_front/check_front.json"
if os.path.exists(p):
    j = json.load(open(p))
    S["e2e_front"] = dict(pass_=j["pass_"], hist=j["hist"], checks=[
        {k: c[k] for k in ("prompt", "k", "suffix_ok", "equal_prefix_cache", "first_diff",
                           "t_req1_s", "t_req2_s", "load_wall_s", "load_pipe_s", "t_ref1_s",
                           "req2_overhead_s")} for c in j["checks"]])

json.dump(S, open(f"{O}/summary.json", "w"), indent=1, default=str)

# markdown
L = ["# 8-GPU restore test summary", "", "## Stages", ""]
for k, v in S["stages"].items():
    L.append(f"- {k}: **{v['status']}** {v['note']}")
L += ["", "## Machine", ""] + [f"- {g}" for g in S.get("gpus", [])]
for k in ("data_dir_fstype", "data_dir_free_gib", "kernel"):
    if k in S.get("preflight", {}):
        L.append(f"- {k}: {S['preflight'][k]}")
if F:
    L += ["", "## fio (GB/s)", ""] + [f"- {k}: {v:.2f}" for k, v in F.items()]


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else "-"


for name, key in (("Concurrent restores (stage 20)", "concurrent"),
                  ("Under write-back (stage 21)", "writeback"),
                  ("SRPT + recompute-front, emulated (stage 31)", "front"),
                  ("Poisson arrivals, response time from arrival (stage 32)", "arrivals")):
    if not S[key]:
        continue
    L += ["", f"## {name}", "", "| exp | G | policy | reps | mean s | max s | agg GB/s | srpt/fair |",
          "|---|---|---|---|---|---|---|---|"]
    for exp, gd in sorted(S[key].items()):
        for gk, d in sorted(gd.items(), key=lambda x: int(x[0][1:])):
            for pol in ("fair", "srpt"):
                if pol in d:
                    s = d[pol]
                    sp_ = d.get("srpt_vs_fair_mean_speedup") if pol == "srpt" else None
                    L.append(f"| {exp} | {gk[1:]} | {pol} | {s['reps']} | {fmt(s['mean_s'])} | "
                             f"{fmt(s['max_s'])} | {fmt(s['agg_GBps'])} | {fmt(sp_)} |")
if "front_schedule" in S:
    L += ["", "## Front schedule and model prediction", ""]
    for g, d in S["front_schedule"].items():
        p_ = d["pred"]
        L.append(f"- {g}: B={d['B_bytes_per_s']/1e9:.1f} GB/s, t(n) {d['t_source']}, k={d['k']}; "
                 f"predicted mean fair {p_['fair']['mean']:.3f} / srpt {p_['srpt']['mean']:.3f} / "
                 f"srpt+front {p_['srpt_front']['mean']:.3f} s")
if S["prefill_s"]:
    L += ["", "## Recompute t(n) (s, median)", ""]
    for m, d in S["prefill_s"].items():
        L.append(f"- {m}: " + ", ".join(f"{n}: {v:.3f}" for n, v in d.items()))
if "e2e_front" in S:
    e = S["e2e_front"]
    L += ["", f"## End-to-end recompute-front (stage 40): pass={e['pass_']}", ""]
    for c in e["checks"]:
        L.append(f"- prompt {c['prompt']} k={c['k']}: suffix_ok={c['suffix_ok']} "
                 f"equal={c['equal_prefix_cache']} req2_overhead_s={c['req2_overhead_s']}")
open(f"{O}/summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
