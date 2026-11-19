#!/usr/bin/env python3
"""Restore-pipeline timeline figure from nsys SQLite exports (job p5).

  nsys export -t sqlite -o copy.sqlite copy_t11_q32.nsys-rep
  python3 plot_timeline.py copy.sqlite [--nooverlap nooverlap.sqlite] -o timeline.pdf

Stages, restricted to the NVTX "run" range (the measured run, not the warmup):
  SSD read  NVTX "read" start/end ranges (one per 512 KiB O_DIRECT read, -DKVR_NVTX build)
  H2D       CUPTI memcpy, copyKind 1
  scatter   CUPTI kernel scatter_kernel; bytes = gridY (items in the batch) x item bytes
Panel (a)/(b): throughput per stage in GB/s, each interval's bytes spread evenly over its
duration, 2 ms bins. Panel (c): 1 ms zoom from the middle of the overlapped run.
"""
import argparse
import sqlite3

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ITEM = 512 << 10  # bytes per read / per scatter item (70B geometry, 512 KiB items)
C = {"read": "#2a78d6", "h2d": "#eb6834", "scatter": "#1baf7a"}  # validated slots 1-3
NAME = {"read": "SSD read", "h2d": "H2D copy", "scatter": "scatter"}
INK, INK2 = "#0b0b0b", "#52514e"


def load(path):
    db = sqlite3.connect(path)
    tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}

    def nvtx(name):
        if "NVTX_EVENTS" not in tabs:
            return np.empty((0, 2))
        q = """select start, end from NVTX_EVENTS
               where coalesce(text, (select value from StringIds where id = textId)) = ?
               and end is not null order by start"""
        return np.array(db.execute(q, (name,)).fetchall(), dtype=float).reshape(-1, 2)

    run = nvtx("run")
    kq = """select k.start, k.end, k.gridY from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on s.id = k.shortName where s.value = 'scatter_kernel'"""
    ker = np.array(db.execute(kq).fetchall(), dtype=float).reshape(-1, 3)
    mq = "select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 1"
    h2d = np.array(db.execute(mq).fetchall(), dtype=float).reshape(-1, 3)
    rd = nvtx("read")
    if len(run):
        t0, t1 = run[-1]
    else:  # no NVTX (old trace): last 90% of the kernel span
        t0, t1 = ker[:, 0].min(), ker[:, 1].max()
        print("warning: no NVTX run range, using kernel span")

    def clip(a):
        return a[(a[:, 0] >= t0) & (a[:, 1] <= t1)] if len(a) else a

    rd = clip(rd)
    rd = np.column_stack([rd, np.full(len(rd), ITEM)]) if len(rd) else np.empty((0, 3))
    ker = clip(ker)
    ker = np.column_stack([ker[:, :2], ker[:, 2] * ITEM]) if len(ker) else np.empty((0, 3))
    st = {"read": rd, "h2d": clip(h2d), "scatter": ker}
    # the NVTX run range also covers setup (thread start, event pool); time 0 = first read
    # submitted (or first copy), end = last scatter done, which is what restore.cu reports
    t0 = min(v[:, 0].min() for v in st.values() if len(v))
    t1 = max(v[:, 1].max() for v in st.values() if len(v))
    for k in st:  # ns -> ms relative to restore start
        st[k] = st[k].copy()
        st[k][:, :2] = (st[k][:, :2] - t0) / 1e6
    dur = (t1 - t0) / 1e6
    print(f"{path}: run {dur:.1f} ms; " + ", ".join(
        f"{k} {len(v)} ops {v[:, 2].sum() / 2**30:.2f} GiB" for k, v in st.items()))
    return st, dur


def rate(iv, dur, bin_ms=2.0, fine_ms=0.01):
    """GB/s per bin, each interval's bytes spread evenly over [start, end]."""
    n = int(np.ceil(dur / fine_ms)) + 1
    acc = np.zeros(n + 1)
    if len(iv):
        s = np.clip((iv[:, 0] / fine_ms).astype(int), 0, n)
        e = np.clip(np.maximum((iv[:, 1] / fine_ms).astype(int), s + 1), 0, n)
        r = iv[:, 2] / (e - s)  # bytes per fine step
        np.add.at(acc, s, r)
        np.add.at(acc, e, -r)
    per_fine = np.cumsum(acc)[:n]
    k = int(round(bin_ms / fine_ms))
    m = n // k
    b = per_fine[: m * k].reshape(m, k).sum(1) / (bin_ms * 1e-3) / 1e9
    return (np.arange(m) + 0.5) * bin_ms, b


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, width=0.6, length=2.5)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)


def throughput(ax, st, dur, xmax, title):
    ymax = 0
    for k in ("read", "h2d", "scatter"):
        x, y = rate(st[k], dur)
        ax.plot(x, y, color=C[k], lw=1.2, solid_capstyle="round", label=NAME[k])
        ymax = max(ymax, y.max() if len(y) else 0)
    ax.axvline(dur, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(dur, ymax * 1.18, f" {dur:.0f} ms", fontsize=6.5, color=INK2, va="top")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax * 1.25)
    ax.set_ylabel("GB/s", color=INK2)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=2)
    style(ax)


def gantt(ax, st, dur, win=1.0):
    a = dur / 2 - win / 2
    b = a + win
    lanes = [("read", 2), ("h2d", 1), ("scatter", 0)]
    for k, y in lanes:
        iv = st[k]
        iv = iv[(iv[:, 1] > a) & (iv[:, 0] < b)]
        if k == "read":  # hundreds of reads overlap: draw reads in flight as a filled step
            t = np.linspace(a, b, 2001)
            inflight = ((iv[:, 0][None, :] <= t[:, None]) & (iv[:, 1][None, :] > t[:, None])).sum(1)
            mx = max(inflight.max(), 1)
            ax.fill_between(t - a, y - 0.35, y - 0.35 + 0.7 * inflight / mx, step="mid",
                            color=C[k], lw=0)
            ax.text(win * 1.01, y, f"≤{mx} in flight", fontsize=6, color=INK2, va="center")
        else:
            ax.broken_barh([(s - a, e - s) for s, e, _ in iv], (y - 0.35, 0.7),
                           facecolors=C[k], edgecolors="white", linewidth=0.3)
    ax.set_yticks([y for _, y in lanes], [NAME[k] for k, _ in lanes], fontsize=6.5)
    ax.set_xlim(0, win)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel(f"time within 1 ms window at t = {a:.0f} ms (ms)", color=INK2)
    style(ax)
    ax.grid(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("overlap")
    p.add_argument("--nooverlap")
    p.add_argument("-o", "--out", default="timeline.pdf")
    a = p.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42,
                         "axes.titlesize": 7.5, "axes.labelsize": 7})
    st, dur = load(a.overlap)
    no = load(a.nooverlap) if a.nooverlap else None
    xmax = max(dur, no[1] if no else 0) * 1.08
    rows = [1, 1, 0.9] if no else [1, 0.9]
    fig, axs = plt.subplots(len(rows), 1, figsize=(3.33, 1.15 * sum(rows) + 0.3),
                            gridspec_kw={"height_ratios": rows}, constrained_layout=True)
    throughput(axs[0], st, dur, xmax, "(a) Overlapped pipeline")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False,
                  fontsize=6.5, handlelength=1.4, columnspacing=1.2, labelcolor=INK)
    if no:
        throughput(axs[1], no[0], no[1], xmax, "(b) No overlap: read all, then copy and scatter")
        axs[1].set_xlabel("time since restore start (ms)", color=INK2)
    else:
        axs[0].set_xlabel("time since restore start (ms)", color=INK2)
    gantt(axs[-1], st, dur)
    axs[-1].set_title(f"({'c' if no else 'b'}) Zoom: stages overlap", fontsize=7.5, color=INK,
                      loc="left", pad=2)
    fig.savefig(a.out)
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=300)  # preview
    print("wrote", a.out)


if __name__ == "__main__":
    main()
