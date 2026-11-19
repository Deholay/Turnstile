# Running guide: 8-GPU shared-SSD KV-restore test

This bundle measures what happens when several GPUs on **one node** restore LLM KV caches
from **the same local NVMe array** at the same time. It also measures whether coordinating
them helps. Read this file completely before running anything, then read `output_guide.md`.

If you are an AI agent operating this bundle for a human, follow the rules in
"Rules for the operator" exactly. Ask the human whenever this guide says to ask.

## 1. What is being tested

Each GPU restores one 32,768-token KV cache from a file on the NVMe array into GPU memory:
- `70b` geometry: 10 GiB per file (Llama-3.3-70B, 80 layers).
- `8b` geometry: 4 GiB per file (Llama-3.1-8B, 32 layers).

The restore program (`src/restore.cu`) works as follows:
- It reads 512 KiB items with O_DIRECT through Linux AIO, using several threads.
- It copies them to the GPU and scatters them into a paged layout.
- It then verifies every word, so a wrong byte fails the run.

Restore policies compared:
- **fair**: every GPU restores on its own. The array's bandwidth is shared equally. This is
  the baseline.
- **srpt**: the processes share a small table in POSIX shared memory (`/dev/shm`). Only the
  restore with the fewest bytes left may issue reads, so restores finish one after another.
- **srpt + recompute-front** (stage 31): restore *i* first spends time `t(k_i)` recomputing
  the first `k_i` tokens on its own GPU. It then loads only tokens `[k_i, 32768)`. In this
  stage the recompute is **emulated by a delay**: the GPU is idle during the delay. The
  delay equals the recompute time measured in stage 30.

The headline numbers are the mean and max completion time of G simultaneous restores, for
G = 1, 2, 4, 8.

## 2. Bundle layout

```
README.md            entry point
running_guide.md     this file
output_guide.md      what to hand back
config.env           the ONLY file the operator must edit
run_1gpu.sh          job 1 (1 GPU): build, vLLM stages 30 and 40
run_8gpu.sh          job 2 (all GPUs): stages 00-32 in priority order, time-limit aware, always packs
job_1gpu.sbatch      Slurm template for job 1 (fill in account/partition)
job_8gpu.sbatch      Slurm template for job 2 (fill in account/partition)
src/                 C++/CUDA restore pipeline + vLLM connector (do not change semantics)
scripts/             one script per stage + launch.py, front_schedule.py, summarize.py
```

Stages (each writes `out/<stage>/` and `out/<stage>/status.json`):

| Stage | What | Needs | Gate? | Rough time |
|---|---|---|---|---|
| Stage | What | Needs | Gate? | Job | Rough time |
|---|---|---|---|---|---|
| `00_preflight` | Record hardware and software; check prerequisites | — | yes | 8-GPU | < 1 min |
| `01_build` | Compile `build/restore` and `build/libkvr.so` with nvcc | CUDA toolkit | yes | both | 1 min |
| `02_smoke` | Word-exact correctness checks, partial restores, 2-GPU SRPT check; writes the 8b files | — | yes | 8-GPU | 1–2 min |
| `20_concurrent` | **Main result**: G = 1, 2, 4, 8 × fair/srpt × 8b/70b × REPS; writes the 70b files | — | no | 8-GPU | 4–7 min |
| `31_front_emul` | SRPT + recompute-front schedule and emulation | stage 20 (stage 30 strongly preferred) | no | 8-GPU | 1–2 min |
| `21_writeback` | Same at G = all GPUs with a background writer | fio (optional) | no | 8-GPU | 1–3 min |
| `32_arrivals` | Poisson arrivals, fair vs srpt | stage 20 | no | 8-GPU | 2–5 min |
| `10_fio` | Raw array bandwidth; read while writing | fio (optional) | no | 8-GPU | 1–2 min |
| `30_prefill` | Recompute time t(n) with vLLM, 1k–32k tokens, 8B (70B if the GPU has ≥ 120 GB) | vLLM env + models | no | 1-GPU | 3–5 min (8B) |
| `40_e2e_front` | End-to-end recompute-front through vLLM on 1 GPU (correctness + overhead) | vLLM env + 8B model | no | 1-GPU | 3–5 min |
| `99_pack` | `summary.md`, `summary.json`, results tarball | — | — | both | < 1 min |

The 8-GPU table rows are in the priority order that `run_8gpu.sh` uses. Each process runs all
REPS repetitions, so process start-up (CUDA initialization, allocating 10 GiB) is paid only
once per configuration.

## 2a. Compute budget (SU)

The whole test is sized for about **200 SU**, assuming the course cluster's rate of about
**60 SU per GPU-hour** (H200). SU = GPUs × hours × rate. If this cluster's rate differs,
recompute the numbers and **ask the human before submitting anything**.

| Job | GPUs | Time limit | Max SU at 60/GPU-h | Expected |
|---|---|---|---|---|
| job 1: `job_1gpu.sbatch` (`run_1gpu.sh`, 8B only) | 1 | 12 min | 12 | 6–10 |
| job 2: `job_8gpu.sbatch` (`run_8gpu.sh`) | 8 | 23 min | 184 | 130–180 |
| **Total** | | | **196** | about 140–190 |

Rules:
- Submit job 1 first: it is cheap and checks that the code builds and the vLLM path works.
  If it fails at build, fix the build before spending SU on job 2.
- Job 2 must stay at `-t 00:23:00` or less. `run_8gpu.sh` skips lower-priority stages when
  the time left is too short, and on the scheduler's warning signal (`--signal=B:USR1@150`)
  it stops and packs. A killed job therefore still leaves a tarball.
- Do **not** add the 70B model to job 1 within this budget. It costs about 15 more SU.
  Do it only if the human approves.
- Do not resubmit job 2 after a failure without asking the human. Report first.
- Waiting in the queue costs no SU. An 8-GPU (whole-node) job can wait a long time: on the
  course cluster, an 8-GPU job was estimated to start about 8–9 hours after submission,
  and one waited more than a day without starting. Submit job 2 early and keep its time
  limit short, so the scheduler can backfill it.

## 3. Requirements

Hardware:
- Up to 8 NVIDIA GPUs in **one node**, each with at least 16 GB of memory. The 70b
  geometry needs about 11 GB of free GPU memory per process.
- A local NVMe device or md RAID with an ext4 or xfs filesystem, mounted where O_DIRECT works.
- At least **180 GiB free** on that filesystem.
- The GPUs and the array must be **idle** during the run. Other jobs make the numbers
  meaningless.

Software:
- Linux, bash, python3 (standard library only for the core stages), `nvidia-smi`.
- CUDA toolkit with `nvcc`, version 12.x or newer, matching the driver.
- Optional: `fio` (stages 10 and 21) and `numactl` (NUMA binding).
- Optional, for stages 30 and 40: a separate Python environment with **vllm==0.10.1.1**.
  The connector in `src/kvr_connector.py` uses that version's KV-connector API.
  ```bash
  python3 -m venv /path/to/venv_kvr
  /path/to/venv_kvr/bin/pip install vllm==0.10.1.1 transformers==4.55.4
  ```
  Transformers 5.x breaks vLLM 0.10. If vLLM 0.10.1.1 cannot run on these GPUs (for
  example, a CUDA mismatch), skip stages 30 and 40 and record why. Do not substitute
  another vLLM version for stage 40.
- Optional models, as local directories (Hugging Face snapshots):
  - `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` (about 9 GB): stages 30 and 40.
  - `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` (about 70 GB): stage 30 only, and only
    on GPUs with at least 120 GB of memory.

  These are Llama-licensed. If a download asks for a license acceptance or a token, **ask
  the human**. Do not accept licenses or use credentials on your own. Put model files
  outside `DATA_DIR`.

No root or sudo is needed. Nothing is installed system-wide by the scripts.

## 4. Setup

1. Unpack the bundle somewhere on local disk. Keep an untouched copy for diffs later, then
   `cd` into the working copy:
   ```bash
   cp -r bundle_8gpu bundle_8gpu.orig
   ```
2. Edit `config.env`:
   - `DATA_DIR`: a new, empty directory on the NVMe array. Everything large is written
     only here.
   - `GPUS`: the physical GPU indices to use, normally `0 1 2 3 4 5 6 7`.
   - `VLLM_PY`, `MODEL_8B`, `MODEL_70B`: fill these in if available; otherwise leave them
     empty.
   - Leave the other values at their defaults unless the human asks otherwise.
3. If the machine uses Slurm, use `job_1gpu.sbatch` and `job_8gpu.sbatch` (section 5).
   Job 2 must hold all 8 GPUs of **one** node, because the array under test is node-local.
   Request the whole node if possible, so nobody else uses the array. Ask the human for the
   account and partition. Inside an allocation, GPU indices are the ones `nvidia-smi` shows
   there, usually 0..n-1. `-c 96` assumes 12 CPUs per GPU; lower it if the cluster limits
   CPUs per GPU, and record the change. If `numactl` binding is refused there, that is fine.
4. Check that nobody else is using the GPUs:
   ```bash
   nvidia-smi
   ```

## 5. Running

With Slurm (fill in `<ACCOUNT>` and `<PARTITION>` first; ask the human for them):

```bash
sbatch job_1gpu.sbatch
```

After job 1 finishes and `out/30_prefill/status.json` is `ok` (or a skip you understand):

```bash
sbatch job_8gpu.sbatch
```

Both jobs write into the same `out/` directory. Job 2 packs everything at the end.

Without a scheduler, on a machine you have to yourself:

```bash
bash run_1gpu.sh 2>&1 | tee out_console_1gpu.log
```

```bash
TIME_LIMIT_MIN=23 bash run_8gpu.sh 2>&1 | tee out_console_8gpu.log
```

To debug a single stage on a machine you already hold (not inside a paid allocation unless
the human agrees), run the stages one by one:

```bash
bash scripts/00_preflight.sh
```

```bash
bash scripts/01_build.sh
```

```bash
bash scripts/02_smoke.sh
```

Continue with `10_fio`, `20_concurrent`, `21_writeback`, `30_prefill`, `31_front_emul`,
`32_arrivals` and `40_e2e_front`, then finish with:

```bash
bash scripts/99_pack.sh
```

Each stage appends to `out/run.log` and writes `out/<stage>/status.json` with status `ok`,
`fail` or `skip`. Re-running a stage appends to its CSV. To redo a stage cleanly, delete
`out/<stage>/` first. The restore files in `DATA_DIR` are written once and reused; their
`*.ok` markers say they are complete.

### Priority if time or access is short

`run_8gpu.sh` already runs the stages in priority order: 00 → 01 → 02 → **20** → 31 → 21 →
32 → 10 → 99. Stage 20 is the main result. Stage 31 uses the measured t(n) only if job 1
ran first; otherwise it falls back to H200 numbers and labels them as such. Always finish
with 99, so partial results are still packed.

## 6. Gates and stop rules

- **00 fails**: fix the configuration and run again. Typical causes: wrong `DATA_DIR`,
  tmpfs or a network filesystem, not enough space, or busy GPUs. Do not continue on a
  network filesystem or tmpfs.
- **01 fails**: see "Changes the operator may make" below.
- **02 fails**: STOP. Any `VERIFY FAILED`, any non-zero `mismatches`, or
  `SRPT_SERIALIZES NO` means the measurements would be invalid. Report it to the human with
  the last 50 lines of `out/02_smoke/smoke.log`. Do not continue to stage 20.
- Any multi-GPU launch reporting `process N not ready` or a non-zero exit: a few failures
  are tolerated, because the stage continues and the summary counts only complete
  repetitions. If more than 20% of launches in a stage fail, stop and report.
- A GPU at 100% utilization by a foreign process, or another user's I/O on the array:
  stop, tell the human, and resume later. Note it in the report.

## 7. Rules for the operator (human or agent)

- Write only inside `DATA_DIR`, the bundle directory, and the models and venv directories
  the human agreed to.
- No sudo, no system configuration changes, no kernel or driver changes. Do not drop page
  caches or change I/O schedulers.
- Do not run anything else on these GPUs or this array while stages 10–40 run.
- Do not change measurement semantics. Specifically, do not change:
  - restore sizes, geometry, number of repetitions, or the timing method;
  - verification;
  - what counts as completion time.
- Do not tune `KVR_THREADS`, `KVR_QD` or `KVR_MODE` per stage. You may run one extra,
  separately labeled sweep if the human asks.
- Do not invent or interpolate missing results. A missing stage stays missing in the report.
- Never put tokens, passwords or usernames into result files.

### Changes the operator may make

The code in `src/` was developed on an H200 node (CUDA 12.6, Linux 4.18, vLLM 0.10.1.1).
Parts of it (partial restore, SRPT coordinator, recompute-front connector path) have
**not yet run on a multi-GPU machine**. You may fix:

- build errors (compiler flags, missing includes, architecture flags in `scripts/01_build.sh`);
- crashes and obvious bugs that prevent a stage from running;
- environment-specific paths.

For every such change:
1. Keep the change minimal and keep the measured behavior identical.
2. Save a patch:
   ```bash
   diff -ru <original_bundle_dir> <your_bundle_dir> > out/bundle_changes/operator.patch
   ```
   Or keep a copy of the original files to diff against.
3. Describe each change in one line in `out/bundle_changes/NOTES.md`.

If a fix would change what is measured, do not make it: report the problem instead.

## 8. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `io_setup: Resource temporarily unavailable` | `fs.aio-max-nr` too low for 8 processes × threads × QD | Lower `KVR_THREADS` to 4, re-run from stage 02, and record it. Do not change sysctl without the human. |
| `O_DIRECT read/write failed` in preflight | Filesystem without O_DIRECT (tmpfs, some network filesystems) | Choose another `DATA_DIR` on local NVMe. |
| `cudaHostAlloc` or out-of-memory errors | Pinned memory limits (`ulimit -l`) or a GPU in use | Check `ulimit -l` and `nvidia-smi`; report. |
| `numactl ... failed` | cgroup or scheduler blocks binding | Nothing to do: `launch.py` silently runs without binding. |
| `process i not ready` | Process crashed during setup | See `out/<stage>/logs/*` and the error printed by `launch.py`. |
| fio `--rate` error | fio version without `m` suffix | Use `--rate=<bytes>` instead; record it. |
| `SRPT_SERIALIZES NO` | Coordinator not working (shared memory not shared, or the grant is ignored) | Stop; report `out/02_smoke/coord2.csv` and the logs. |
| vLLM import or CUDA error in stage 30 or 40 | vLLM 0.10.1.1 does not match this CUDA or driver | Skip 30/40, record the error. Stage 31 then uses fallback t(n) and labels it. |
| stage 40 `RESULT FAIL` | A correctness problem in the recompute-front path | Do not retry with changes; pack and report. This is a finding. |

## 9. When done

1. `run_8gpu.sh` runs `scripts/99_pack.sh` itself. Run it by hand only after a manual run. Add `--clean` only if the human wants the 112 GiB of
   restore files removed from `DATA_DIR`.
2. Follow `output_guide.md` to check and report.
