#!/usr/bin/env python3
"""Schedule figure (fig:schedule): G=4 simultaneous 32k-token 8B restores under fair sharing,
SRPT, and SRPT + recompute-front, drawn to scale from the bandwidth model of Sec. 3.2.

  python3 analysis/plot_schedule.py -o ppopp-template/figures/schedule.pdf

Same inputs as analysis/front_model.py: B = 54.2 GB/s, measured 8B prefill t(n), linearly
interpolated; k is searched over 128-token segment boundaries (Eq. front).
"""
import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

B = 54.2e9
G, N, SEG, S_TOK = 4, 32768, 128, 131072  # 8B: 128 KiB per token
PTS = np.array([0, 4096, 8192, 16384, 32768])
T8 = np.array([0, 0.067, 0.150, 0.370, 0.996])  # measured 8B prefill (s), as in front_model.py


def t(n):
    return float(np.interp(n, PTS, T8))


def schedules():
    S = N * S_TOK
    fair = [[("load", 0.0, G * S / B)] for _ in range(G)]
    srpt = [[("wait", 0.0, i * S / B), ("load", i * S / B, (i + 1) * S / B)] for i in range(G)]
    front, W = [], 0.0
    for _ in range(G):
        f, k = min((max(W, t(k)) + (N - k) * S_TOK / B, k)
                   for k in range(0, N + 1, SEG))
        start = max(W, t(k))
        segs = [("recompute", 0.0, t(k)), ("wait", t(k), start), ("load", start, f)]
        front.append([s for s in segs if s[2] - s[1] > 0.002])  # drop sub-2 ms slivers
        W = f
    return fair, srpt, front


C = {"load": "#eb6834", "recompute": "#1baf7a"}
INK, INK2 = "#0b0b0b", "#52514e"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", default="schedule.pdf")
    a = p.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fair, srpt, front = schedules()
    panels = [("(a) Fair sharing, each at $B/4$", fair),
              ("(b) SRPT, one at a time at $B$", srpt),
              ("(c) SRPT + recompute-front", front)]
    fig, axs = plt.subplots(3, 1, figsize=(3.33, 2.95), sharex=True, constrained_layout=True)
    for ax, (title, sched) in zip(axs, panels):
        ends = []
        for i, segs in enumerate(sched):
            y = G - 1 - i
            for kind, s, e in segs:
                if kind == "wait":
                    ax.barh(y, e - s, left=s, height=0.62, color="none", ec="#b9b7b2",
                            lw=0.6, ls=(0, (2, 1.5)))
                else:
                    alpha = 0.45 if sched is fair else 1.0
                    ax.barh(y, e - s, left=s, height=0.62, color=C[kind], alpha=alpha, lw=0)
            ends.append(segs[-1][2])
        mean = sum(ends) / len(ends)
        ax.axvline(mean, color=INK, lw=0.7, ls="--")
        ax.set_title(f"{title}; mean {mean * 1e3:.0f} ms", loc="left", fontsize=8, pad=2,
                     color=INK)
        ax.set_yticks(range(G), [f"GPU{G - 1 - y}" for y in range(G)], color=INK2)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.5, G - 0.2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        print(title, "mean %.3f max %.3f" % (mean, max(ends)))
    axs[-1].set_xlim(0, 0.33)
    axs[-1].set_xlabel("time since the restores arrive (s)", color=INK2)
    fig.legend(handles=[Patch(color=C["load"], label="load (array)"),
                        Patch(color=C["recompute"], label="recompute (GPU)"),
                        Patch(fc="none", ec="#b9b7b2", ls="--", label="wait")],
               loc="outside upper center", ncol=3, frameon=False, fontsize=8,
               handlelength=1.2, columnspacing=0.8)
    fig.savefig(a.o)
    fig.savefig(a.o.rsplit(".", 1)[0] + ".png", dpi=300)
    print("wrote", a.o)


if __name__ == "__main__":
    main()
