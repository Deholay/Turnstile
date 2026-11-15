#!/usr/bin/env python3
"""Kill test verdict: SSD read time of a history's KV vs. recomputing it (pass if read < 0.5 x prefill)."""
import csv
import glob
import json
import os
import statistics
import sys

PASS_RATIO = 0.5


def med_iqr(xs):
    q1, q2, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return q2, q3 - q1


def main(out):
    bws = [json.load(open(p))["jobs"][0]["read"]["bw_bytes"]
           for p in sorted(glob.glob(os.path.join(out, "fio-run-*.json")))]
    if not bws:
        sys.exit(f"no fio-run-*.json in {out}")
    bw, bw_iqr = med_iqr(bws)
    meta = json.load(open(os.path.join(out, "prefill_meta.json")))
    h2d = meta["h2d_pinned_bytes_per_s"]

    walls, kv = {}, {}
    with open(os.path.join(out, "prefill.csv")) as f:
        for r in csv.DictReader(f):
            n = int(r["tokens"])
            kv[n] = int(r["kv_bytes"])
            if r["phase"] == "run":
                walls.setdefault(n, []).append(float(r["wall_s"]))

    cols = ["tokens", "prefill_median_s", "prefill_iqr_s", "kv_gib", "nvme_read_s",
            "read_over_prefill", "read_plus_h2d_s", "verdict"]
    rows = []
    for n in sorted(walls):
        p, p_iqr = med_iqr(walls[n])
        read = kv[n] / bw
        ratio = read / p
        rows.append([n, f"{p:.4f}", f"{p_iqr:.4f}", f"{kv[n] / 2**30:.3f}", f"{read:.4f}",
                     f"{ratio:.3f}", f"{read + kv[n] / h2d:.4f}",
                     "PASS" if ratio < PASS_RATIO else "FAIL"])

    with open(os.path.join(out, "summary.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    print(f"fio seq read (bs=1M, QD32, O_DIRECT): median {bw / 1e9:.2f} GB/s, IQR {bw_iqr / 1e9:.2f} GB/s, "
          f"{len(bws)} runs")
    print(f"pinned H2D: {h2d / 1e9:.2f} GB/s; KV/token {meta['kv_bytes_per_token_formula'] / 1024:.0f} KiB")
    print("\t".join(cols))
    for r in rows:
        print("\t".join(map(str, r)))


if __name__ == "__main__":
    main(sys.argv[1])
