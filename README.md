# Turnstile: coordinated KV-cache restore from a shared SSD array

Code, job scripts, raw results and paper source for restoring multi-turn LLM KV caches from a
node-local NVMe array instead of recomputing them, and for coordinating concurrent restores
across up to 8 GPUs that share one array (fair sharing vs. SRPT, plus recompute-front).

Every number in the paper comes from a result directory in this repo. Each result directory
holds a `commands.txt` with the exact command lines, the node, the GPU driver, package versions
and the sha256 of every source file and binary the job used.

## Layout

| Path | What it is |
|---|---|
| `harness/` | Restore pipeline, current version: `kvr_core.cuh` (io_uring/AIO reads → pinned staging → H2D → scatter), `restore.cu` (standalone benchmark and byte-exact check), `kvr_lib.cu` + `kvr_connector.py` (vLLM 0.10.1.1 KV connector), `kvr_coord.h` (cross-process SRPT coordinator), `check_front.py`, `check_greedy.py`, `ttft_kvr.py`, `iotrace.py`, `build.sh`. Used by P5 and P7. |
| `harness/v1115/` | The 2026-11-15 version of the four core sources, which P1, P2, P4 and P6 ran on (sha256 match their `commands.txt`). |
| `harness/p1_job.sh`, `p2_job.sh`, `results/` | P1 (restore pipeline sweep/ablation) and P2 (vLLM TTFT + greedy check), H200, 2026-11-15. |
| `nchc_p4/` | P4: vLLM 0.32 native disk tier attempt, 70B greedy check, 8B recompute and restore at 4k–128k. P4d: the disk-tier crash debug run. |
| `nchc_p6/` | P6: every single-GPU 32k TTFT in one allocation (ours, LMCache default / O_DIRECT / DRAM). |
| `nchc_p7/` | P7: recompute-front end-to-end check through vLLM (8B). Sources = `harness/`. |
| `p5_nsys/` | P5: Nsight Systems restore timelines and `plot_timeline.py`. Sources = `harness/` built with `-DKVR_NVTX`. |
| `bundle_8gpu/` | Self-contained 8-GPU test bundle (stages 00–99, own copy of the sources in `src/`, own guides). `results/8gpu-2026-11-20/` is the 8×H200 run used for the multi-GPU results. |
| `kill-test/` | Feasibility test on an RTX 5090 workstation (fio + HF prefill timing). |
| `analysis/` | `front_model.py` (bandwidth model, Table bounds), `plot_multi.py` (Figure multi). |
| `envs/` | `pip freeze` of the three NCHC Python environments (`kt2`, `kt3`, `kt4`). |
| `ppopp-template/` | Paper source (`make` builds `main.pdf`). |
| `PLAN.zh-TW.md` | Working plan and decision log (Traditional Chinese). |

The source copies in `harness/`, `harness/v1115/` and `bundle_8gpu/src/` differ on purpose:
each is the exact code a set of jobs ran. Do not merge them when reproducing a specific result;
use the copy listed for that result below.

## Paper → result map

| Paper item | Job(s) | Result files | Code |
|---|---|---|---|
| Table `setup` | all | `*/results/*/commands.txt`, `bundle_8gpu/results/8gpu-2026-11-20/00_preflight/` | — |
| Table `motiv` | P0, kill test, job 472017 | `kill-test/results/`; P0 and 472017 are **not in this repo** (see Gaps) | `kill-test/` |
| Table `single` | P6 481748 | `nchc_p6/results/p6-2026-11-19-481748/{ours,lmc}_*.json` | `nchc_p6/`, `harness/v1115/` |
| Table `histlen` | P4 481654, 481677 (+472017 for 4k–32k recompute) | `nchc_p4/results/*/restore_sweep_8b.csv`, `prefill_long_8b.csv` | `nchc_p4/`, `harness/v1115/` |
| Table `ablation` | P1 472200, P5 481692/481703 | `harness/results/p1-*/restore.csv`, `p5_nsys/results/*/restore.csv` | `harness/p1_job.sh`, `p5_nsys/` |
| Figure `timeline` | P5 481703, 481707 | `p5_nsys/results/*/timeline.pdf` (nsys traces not in repo) | `p5_nsys/plot_timeline.py` |
| Figure `multi` | 8-GPU run | `bundle_8gpu/results/8gpu-2026-11-20/{20_concurrent,31_front_emul}/` | `analysis/plot_multi.py` |
| Table `arrivals` | 8-GPU run | `bundle_8gpu/results/8gpu-2026-11-20/32_arrivals/` | `bundle_8gpu/scripts/32_arrivals.sh` |
| Table `bounds` | model | stdout of `analysis/front_model.py` | `analysis/front_model.py` |
| Correctness (greedy equality, checksums) | P2 472211, P4 481677, P7 481829 | `ttft_*.json`, `greedy_70b.json`, `check_front.json` | `ttft_kvr.py`, `check_greedy.py`, `check_front.py` |
| vLLM 0.32 disk tier crashes | P4 481677, P4d 481718 | `nchc_p4/results/*/disk_tier_70b.log`, `p4d-*/d*.log` | `nchc_p4/disk_tier_ttft.py` (version run: git `1a52416`) |

## Regenerate figures, model and paper (no GPU)

Needs Python 3 with numpy and matplotlib, and TeX Live / MacTeX with latexmk.

```bash
python3 analysis/front_model.py
```

```bash
python3 analysis/plot_multi.py bundle_8gpu/results/8gpu-2026-11-20 -o ppopp-template/figures/multi.pdf
```

```bash
make -C ppopp-template
```

`plot_timeline.py` reads the SQLite exports of the nsys traces (`nsys export --type sqlite`),
which are too large for git:
`python3 p5_nsys/plot_timeline.py <overlap>.sqlite --nooverlap <nooverlap>.sqlite -o timeline.pdf`.

## Reproduce the measurements

### Hardware and system

- 1 or 8 NVIDIA H200 (sm_90) on one node; driver 550.127.08 (CUDA ≤ 12.4 runtime API at the
  driver level), CUDA toolkit 12.6 for `nvcc`.
- Node-local NVMe array with a filesystem that supports `O_DIRECT` (we used 8× PM9A3 in md RAID0
  mounted at `/tmp` on NCHC nano5 `normal2` nodes; 8-GPU run: xfs). Not tmpfs, NFS or Lustre.
- Free space on the array: ~30 GiB for a single-GPU job, ≥ 180 GiB for the 8-GPU bundle;
  `numactl`.
- Models (local snapshots): `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`,
  `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`. The kill test used
  `meta-llama/Llama-3.1-8B-Instruct` (BF16).

### Python environments

| Env | Used by | Key packages | Lock file |
|---|---|---|---|
| `kt2` | ours (connector), P2, P4 B/C, P6 ours, P7, bundle stages 30/40 | Python 3.12, vllm 0.10.1.1, torch 2.7.1+cu126, transformers 4.55.4 | `envs/nchc-kt2.txt` |
| `kt3` | P6 LMCache baselines | kt2 + lmcache 0.3.5 | `envs/nchc-kt3.txt` |
| `kt4` | P4 A (vLLM native disk tier) | vllm 0.32.0+cu129, torch 2.13.0+cu129, transformers 5.19.0 | `envs/nchc-kt4.txt` |
| kill test | `kill-test/` | torch 2.11.0+cu128, transformers 5.23.0, fio 3.38 | versions in `kill-test/results/*/env.txt` |

transformers 5.x breaks vLLM 0.10.1.1, so keep `kt2`/`kt3` on 4.55.4. Example:

```bash
python3.12 -m venv kt2 && kt2/bin/pip install vllm==0.10.1.1 transformers==4.55.4
```

`kt4` also needs FlashInfer's precompiled kernels (`vllm download-kernels`).

### Build

```bash
bash harness/build.sh
```

`build.sh` emits sm_90 SASS plus compute_90 PTX. For the 11/15 version, copy
`harness/build.sh` into a directory with the `harness/v1115/` sources and run it there. For the
P5 NVTX build add `-DKVR_NVTX` (and `-ldl` on glibc < 2.34) to the `restore` line.

### Paths the single-GPU job scripts expect

The job scripts were written for NCHC nano5 and are kept as run, because their sha256 is
recorded in each `commands.txt`. To rerun them elsewhere, change the `#SBATCH -A`/`-p` lines and
the path variables at the top of each script (`H`, `P`, `K`, `KT2`–`KT4`, `M8`, `M70`), or
recreate this layout:

| Location on the cluster | Contents |
|---|---|
| `/home/$USER/harness/` | `harness/v1115/*` + `harness/{build.sh,ttft_kvr.py,p1_job.sh,p2_job.sh}`, built, plus `logs/` |
| `/home/$USER/p4/`, `p6/` | `nchc_p4/*.py,*.sh`, `nchc_p6/*.py,*.sh`, plus `logs/` |
| `/home/$USER/p5_nsys/` | `harness/{restore.cu,kvr_core.cuh,kvr_coord.h}` built with `-DKVR_NVTX` + `p5_nsys/*.sh`, plus `logs/` |
| `/home/$USER/p7_front/` | `harness/*` built + `nchc_p7/p7_job.sh`, plus `logs/` |
| `/work/$USER/killtest/envs/{kt2,kt3}`, `/home/$USER/envs/kt4` | Python environments |
| `/home/$USER/models/Meta-Llama-3.1-8B-Instruct-FP8-dynamic`, `/work/$USER/killtest/models/Llama-3.3-70B-Instruct-FP8-dynamic` | models |

Then submit, one GPU each:

```bash
sbatch harness/p1_job.sh
```

(likewise `p2_job.sh`, `nchc_p4/p4_job.sh`, `p5_nsys/p5_job.sh`, `p5_nsys/p5b_job.sh`,
`nchc_p6/p6_job.sh`, `nchc_p7/p7_job.sh`). Every job writes into `<dir>/results/<tag>-<date>-<jobid>/`.

### 8-GPU results

`bundle_8gpu/` is portable and parameterized. Read `bundle_8gpu/running_guide.md`, edit only
`bundle_8gpu/config.env` (`DATA_DIR`, `VLLM_PY`, `MODEL_8B`, `MODEL_70B`) and the
`<ACCOUNT>`/`<PARTITION>` lines of `job_8gpu.sbatch`, then:

```bash
cd bundle_8gpu && sbatch job_8gpu.sbatch
```

Output lands in `bundle_8gpu/out/` (git-ignored); `output_guide.md` explains each file. The
archived run `results/8gpu-2026-11-20/` records the operator's `config.env` and the sha256 of
the bundle at run time in `bundle_changes/`. The operator edited two files on the node
(`scripts/01_build.sh`, a gcc fix, and `src/check_front.py`; the repo keeps the shipped
originals); every other source and script matches the repo. Stages 10 and 21 were skipped there
(no fio on the node), stage 40 ran separately as P7.

### Kill test (RTX 5090 workstation)

`kill-test/run_kill_test.sh` hardcodes its data root in `D=` at the top; set it to a directory
on the NVMe under test with `venvs/kvssd`, `tools/bin/fio` and `models/Llama-3.1-8B-Instruct`
under it. It refuses to start while another process uses the GPU.

## Gaps

- P0 (LMCache time breakdown) and job 472017 (8B recompute at 4k–32k) predate this repo; their
  numbers in Table `motiv`/`histlen` are recorded only in `PLAN.zh-TW.md`.
- P1 (472200) ran a slightly earlier `restore.cu` (sha256 `ebdf20cf…`) than `harness/v1115/`;
  that exact file is lost. P4 and later used `harness/v1115/restore.cu`.
- nsys `.nsys-rep`/`.sqlite` traces and P4's raw `diskstats.csv` (~40 MB) are not in git.
