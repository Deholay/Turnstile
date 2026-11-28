#!/usr/bin/env python3
"""Launch G restore processes that share the SSD array and run REPS repetitions, each
repetition started on a common clock. Writes one CSV row per process per repetition.

Every process is started once with `--warmup 0 --runs REPS --go-file DIR/goI`. Before run r it
prints `READY pid P run r` and waits for the file `DIR/goI.r`. For each repetition the launcher
waits for READY from all processes, picks T0 on CLOCK_MONOTONIC, and creates goI.r at
T0 + delay(i, r). The restore binary prints `timing pid P run r bytes N start S end E`
(CLOCK_MONOTONIC) and then `run r: ... mismatches M` after verification, so completion time
of process i in repetition r is E - T0 for every policy. The next repetition starts only
after every process has finished verifying the previous one.

  python3 launch.py --spec spec.json --csv rows.csv --exp NAME --reps 5 --policy fair|srpt \
      [--restore build/restore] [--threads 8 --qd 32 --mode copy] [--ready-timeout 180]

spec.json: list of {"gpu": int, "file": path, "geo": "70b"|"8b",
                    "start_tokens": int (default 0),
                    "delay_s": float, or list of REPS floats (default 0)}
policy srpt passes the same --coord name to every process (shared-memory SRPT grant).
Exit code 0 only if every process completes every repetition with 0 mismatches.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import json

GEO = {"70b": ["--layers", "80", "--kv-heads", "8", "--head-dim", "128", "--tokens", "32768"],
       "8b": ["--layers", "32", "--kv-heads", "8", "--head-dim", "128", "--tokens", "32768"],
       # P8: short histories for the G=8 load-vs-recompute crossover
       "8b4k": ["--layers", "32", "--kv-heads", "8", "--head-dim", "128", "--tokens", "4096"],
       "8b8k": ["--layers", "32", "--kv-heads", "8", "--head-dim", "128", "--tokens", "8192"]}
RE_READY = re.compile(r"^READY pid \d+ run (\d+)")
RE_TIMING = re.compile(r"^timing pid \d+ run (\d+) bytes (\d+) start ([\d.]+) end ([\d.]+)")
RE_RUN = re.compile(r"^run (\d+): .*mismatches (\d+)")


def numa_prefix(gpu):
    try:
        bus = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            text=True).strip().lower()
        node = None
        for b in (bus[4:] if bus.startswith("0000") else bus, bus):
            p = f"/sys/bus/pci/devices/{b}/numa_node"
            if os.path.exists(p):
                node = int(open(p).read().strip())
                break
        if node is None or node < 0 or not shutil.which("numactl"):
            return []
        pre = ["numactl", f"--cpunodebind={node}", f"--membind={node}"]
        if subprocess.run(pre + ["true"], capture_output=True).returncode == 0:
            return pre
    except Exception:
        pass
    return []


class Proc:
    def __init__(self, cmd, env):
        self.cmd = cmd
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  env=env, text=True, bufsize=1)
        self.lines, self.ready, self.timing, self.result = [], set(), {}, {}
        self.done = False
        self.cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for ln in self.p.stdout:
            with self.cv:
                self.lines.append(ln)
                if m := RE_READY.match(ln):
                    self.ready.add(int(m.group(1)))
                elif m := RE_TIMING.match(ln):
                    self.timing[int(m.group(1))] = (int(m.group(2)), float(m.group(3)),
                                                   float(m.group(4)))
                elif m := RE_RUN.match(ln):
                    self.result[int(m.group(1))] = int(m.group(2))
                self.cv.notify_all()
        self.p.wait()
        with self.cv:
            self.done = True
            self.cv.notify_all()

    def wait_for(self, pred, deadline):
        with self.cv:
            while not pred() and not self.done:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.cv.wait(left)
            return pred()

    def tail(self):
        return "".join(self.lines)[-2000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--policy", choices=["fair", "srpt"], required=True)
    ap.add_argument("--restore", default=os.path.join(os.path.dirname(__file__), "..", "build",
                                                      "restore"))
    ap.add_argument("--threads", default="8")
    ap.add_argument("--qd", default="32")
    ap.add_argument("--mode", default="copy")
    ap.add_argument("--ready-timeout", type=float, default=180)
    ap.add_argument("--run-timeout", type=float, default=120)
    ap.add_argument("--logdir", default=None, help="keep per-process stdout here")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    G = len(spec)

    def delay(i, r):
        d = spec[i].get("delay_s", 0.0)
        return float(d[r] if isinstance(d, list) else d)

    tmp = tempfile.mkdtemp(prefix="kvr_launch_")
    coord = f"kvr_{os.getpid()}" if a.policy == "srpt" else None
    numa, procs = {}, []
    for i, s in enumerate(spec):
        g = s["gpu"]
        if g not in numa:
            numa[g] = numa_prefix(g)
        cmd = numa[g] + [a.restore, "--file", s["file"], *GEO[s["geo"]],
                         "--threads", a.threads, "--qd", a.qd, "--mode", a.mode,
                         "--warmup", "0", "--runs", str(a.reps),
                         "--go-file", os.path.join(tmp, f"go{i}"),
                         "--start-tokens", str(s.get("start_tokens", 0)), "--seed", "1"]
        if coord:
            cmd += ["--coord", "/" + coord]
        procs.append(Proc(cmd, dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))

    fields = ["exp", "rep", "policy", "G", "idx", "gpu", "geo", "start_tokens", "delay_s",
              "bytes", "start_rel_s", "end_rel_s", "restore_s", "mismatches", "exit"]
    fresh = not os.path.exists(a.csv)
    f = open(a.csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if fresh:
        w.writeheader()
    ok = True
    for r in range(a.reps):
        dl = time.monotonic() + (a.ready_timeout if r == 0 else a.run_timeout)
        if not all(p.wait_for(lambda p=p: r in p.ready, dl) for p in procs):
            for i, p in enumerate(procs):
                if r not in p.ready:
                    sys.stderr.write(f"process {i} not ready for run {r}:\n{p.tail()}\n")
            ok = False
            break
        t0 = time.monotonic() + 0.05
        for i in sorted(range(G), key=lambda i: delay(i, r)):
            at = t0 + delay(i, r)
            while (d := at - time.monotonic()) > 0:
                time.sleep(d - 0.001 if d > 0.002 else 0.0002)
            open(os.path.join(tmp, f"go{i}.{r}"), "w").close()
        dl = time.monotonic() + a.run_timeout
        for i, (p, s) in enumerate(zip(procs, spec)):
            got = p.wait_for(lambda p=p: r in p.result, dl)
            row = dict(exp=a.exp, rep=r, policy=a.policy, G=G, idx=i, gpu=s["gpu"], geo=s["geo"],
                       start_tokens=s.get("start_tokens", 0), delay_s=f"{delay(i, r):.6f}",
                       mismatches=p.result.get(r, ""), exit="" if not p.done else p.p.returncode)
            if r in p.timing:
                b, st, en = p.timing[r]
                row.update(bytes=b, start_rel_s=f"{st - t0:.6f}", end_rel_s=f"{en - t0:.6f}",
                           restore_s=f"{en - st:.6f}")
            if not got or r not in p.timing or p.result.get(r) != 0:
                ok = False
                sys.stderr.write(f"process {i} failed in run {r}:\n{p.tail()}\n")
            w.writerow(row)
        f.flush()
        if not ok:
            break
    f.close()

    for p in procs:
        if not ok:
            p.p.kill()
        try:
            p.p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.p.kill()
            ok = False
    if a.logdir:
        os.makedirs(a.logdir, exist_ok=True)
        for i, p in enumerate(procs):
            with open(os.path.join(a.logdir, f"{a.exp}_{a.policy}_G{G}_p{i}.log"), "w") as lf:
                lf.write(" ".join(p.cmd) + "\n" + "".join(p.lines))
    if any(p.p.returncode != 0 for p in procs):
        ok = False
    if coord and os.path.exists(f"/dev/shm/{coord}"):
        os.remove(f"/dev/shm/{coord}")
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
