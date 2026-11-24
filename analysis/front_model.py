"""Bandwidth model: restore policies for G concurrent 32k restores on one shared SSD array.

Inputs are H200 measurements (PLAN.zh-TW.md): B = 54.2 GB/s, 8B prefill t(n) at 0/4k/8k/16k/32k,
linear interpolation in between. 70B t(n) is the measured prefill time of the 8-GPU node (stage 30,
bundle_8gpu/results/8gpu-2026-11-20/30_prefill/prefill.csv, median of 5 runs) at 1k..32k.
Simultaneous arrivals only. Run: python3 analysis/front_model.py
"""
import numpy as np
B=54.2e9; G=8; N=32768; seg=128
pts=np.array([0,4096,8192,16384,32768]); t8=np.array([0,0.067,0.150,0.370,0.996])
def mk(scale, tok_bytes):
    t=lambda n: np.interp(n,pts,t8*scale)
    return t, tok_bytes
p70=np.array([0,1024,2048,4096,6144,8192,12288,16384,20480,24576,28672,32768])
t70=np.array([0,0.133,0.264,0.550,0.873,1.221,1.932,2.894,3.785,4.742,5.848,7.015])
models={'8B':mk(1.0,131072),'70B':(lambda n: np.interp(n,p70,t70),327680)}
ks=np.arange(0,N+1,seg)
for name,(t,s) in models.items():
    S=N*s; tN=t(N)
    fair=G*S/B
    srpt=np.mean([(i+1)*S/B for i in range(G)])
    # fair + load-front hybrid (load prefix at B/G, recompute suffix, layer-overlapped)
    lf=min(max(k*s/(B/G), tN-t(k)) for k in ks)
    # fair + recompute-front, suffix prefetched concurrently into pinned DRAM, then H2D at 46 GB/s
    rf=min(max(t(k),(N-k)*s/(B/G)) + (N-k)*s/46e9*0 for k in ks)  # H2D overlappable? report both
    rf_h2d=min(max(t(k),(N-k)*s/(B/G)) + (N-k)*s/46e9 for k in ks)
    # SRPT + load-front: queued GPU idle while waiting
    W=0; fin=[]
    for i in range(G):
        best=None
        for k in ks:
            ld=k*s/B; f=W+max(ld, tN-t(k))
            if best is None or f<best[0]: best=(f,W+ld)
        fin.append(best[0]); W=best[1]
    srpt_lf=np.mean(fin); srpt_lf_max=max(fin)
    # SRPT + recompute-front: recompute prefix while waiting, then load suffix at full B
    W=0; fin=[]; kk=[]
    for i in range(G):
        best=None
        for k in ks:
            f=max(W,t(k))+(N-k)*s/B
            if best is None or f<best[0]: best=(f,k)
        fin.append(best[0]); kk.append(best[1]); W=best[0]
    print(f"== {name}: recompute {tN:.3f}s, one load {S/B:.3f}s")
    print(f" fair load-only        mean {fair:.3f}  max {fair:.3f}")
    print(f" SRPT load-only        mean {srpt:.3f}  max {fair:.3f}")
    print(f" fair+load-front       mean {lf:.3f}  ({fair/lf:.2f}x vs fair)")
    print(f" fair+recompute-front  mean {rf:.3f} (H2D hidden) / {rf_h2d:.3f} (H2D after)  ({fair/rf:.2f}x / {fair/rf_h2d:.2f}x)")
    print(f" SRPT+load-front       mean {srpt_lf:.3f}  max {srpt_lf_max:.3f}  ({fair/srpt_lf:.2f}x)")
    print(f" SRPT+recompute-front  mean {np.mean(fin):.3f}  max {max(fin):.3f}  ({fair/np.mean(fin):.2f}x)  k={[int(x) for x in kk]}")
print("== serial recompute-front under fair sharing (no prefetch): t(k) then load suffix at B/G")
for name,(t,s) in models.items():
    fair=G*N*s/B
    v=min((t(k)+(N-k)*s/(B/G),k) for k in ks)
    print(f" {name[:3]} mean {v[0]:.3f} k={v[1]} ({fair/v[0]:.2f}x)")
