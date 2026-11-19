"""Device-level I/O trace for MSS: samples /proc/diskstats (no root needed; blktrace needs root).

Writes one CSV row per device per sample; timestamps are time.monotonic() (CLOCK_MONOTONIC),
the same clock as the per-read trace from libkvr / restore --trace, so both line up.
Columns are the raw cumulative kernel counters; rates come from differences between samples.
Runs until SIGTERM/SIGINT.

  python iotrace.py OUT.csv [interval_s=0.05] [device regex=^(nvme\\d+n\\d+|md\\d+)$]
"""
import re
import signal
import sys
import time

out = sys.argv[1]
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
pat = re.compile(sys.argv[3] if len(sys.argv) > 3 else r"^(nvme\d+n\d+|md\d+)$")
cols = ["reads", "reads_merged", "sectors_read", "ms_reading", "writes", "writes_merged",
        "sectors_written", "ms_writing", "in_flight", "ms_io", "ms_weighted"]
stop = False


def on_signal(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)
with open(out, "w") as f:
    f.write("t_mono,dev," + ",".join(cols) + "\n")
    nxt = time.monotonic()
    while not stop:
        t = time.monotonic()
        with open("/proc/diskstats") as d:
            for line in d:
                p = line.split()
                if pat.match(p[2]):
                    f.write(f"{t:.6f},{p[2]}," + ",".join(p[3:3 + len(cols)]) + "\n")
        nxt += dt
        time.sleep(max(0.0, nxt - time.monotonic()))
