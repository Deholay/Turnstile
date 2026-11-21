#!/usr/bin/env python3
"""Multi-GPU restore figure (fig:multi) from the 8-GPU run.

  python3 analysis/plot_multi.py out -o ppopp-template/figures/multi.pdf

Reads <out>/20_concurrent/concurrent.csv (fair, srpt at G = 1, 2, 4, 8) and
<out>/31_front_emul/front.csv (srpt + recompute-front at G = 8). Restore time of a request =
end_rel_s - delay_s, i.e. from its own arrival; for recompute-front the delay is the emulated
recompute of the prefix, so end_rel_s is the full response time. Per repetition: mean and max
over the G requests; plotted: median over the repetitions.
"""
import argparse
import collections
import csv
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"fair": "#2a78d6", "srpt": "#eb6834", "front": "#1baf7a"}
INK, INK2 = "#0b0b0b", "#52514e"


def per_rep(path, front=False):
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        t = float(r["end_rel_s"]) if front else float(r["end_rel_s"]) - float(r["delay_s"])
        g[(r["geo"], r["policy"], int(r["G"]), r["rep"])].append(t)
    out = collections.defaultdict(lambda: ([], []))
    for (geo, pol, G, _), ts in g.items():
        out[(geo, pol, G)][0].append(st.mean(ts))
        out[(geo, pol, G)][1].append(max(ts))
    return {k: (st.median(m), st.median(x)) for k, (m, x) in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("-o", default="multi.pdf")
    a = p.parse_args()
    conc = per_rep(f"{a.out}/20_concurrent/concurrent.csv")
    front = per_rep(f"{a.out}/31_front_emul/front.csv", front=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42,
                         "axes.titlesize": 7.5, "axes.labelsize": 7})
    fig, axs = plt.subplots(1, 2, figsize=(3.33, 1.75), constrained_layout=True)
    Gs = [1, 2, 4, 8]
    recompute = {"8b": 0.957, "70b": 7.015}  # stage 30, 32k prefill on the same node
    for ax, geo, title in zip(axs, ["8b", "70b"], ["(a) 8B", "(b) 70B"]):
        for pol, lab in [("fair", "fair (uncoordinated)"), ("srpt", "SRPT")]:
            xs = [G for G in Gs if (geo, pol, G) in conc or (pol == "srpt" and G == 1)]
            src = lambda G: conc[(geo, "fair" if G == 1 else pol, G)]  # G=1: one restore
            ax.plot(xs, [src(G)[0] for G in xs], "-o", color=C[pol], ms=2.5, lw=1.1,
                    label=lab + ", mean")
            ax.plot(xs, [src(G)[1] for G in xs], "--", color=C[pol], lw=0.9, label=lab + ", slowest")
        m, x = front[(geo, "srpt", 8)]
        ax.plot([8], [m], "D", color=C["front"], ms=3.5, label="SRPT + recompute-front, mean")
        ax.plot([8], [x], "d", mfc="none", color=C["front"], ms=4, label="SRPT + recompute-front, slowest")
        if recompute[geo] < 2:
            ax.axhline(recompute[geo], color=INK2, lw=0.7, ls=":")
            ax.text(1, recompute[geo] * 0.97, "recompute", color=INK2, fontsize=6, va="top")
        else:
            ax.text(1, 1.78, f"recompute: {recompute[geo]:.1f} s", color=INK2, fontsize=6, va="top")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Gs, [str(G) for G in Gs])
        ax.set_xlabel("GPUs restoring at once ($G$)", color=INK2)
        ax.set_title(title, loc="left", pad=2, color=INK)
        ax.set_ylim(0, None)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("restore time (s)", color=INK2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False, fontsize=5.8,
               handlelength=1.8, columnspacing=1.0, labelcolor=INK)
    fig.savefig(a.o)
    fig.savefig(a.o.rsplit(".", 1)[0] + ".png", dpi=300)
    print("wrote", a.o)
    for k in sorted(conc):
        print(k, "mean %.3f max %.3f" % conc[k])
    for k in sorted(front):
        print(k, "mean %.3f max %.3f" % front[k])


if __name__ == "__main__":
    main()
