# Output guide: what to hand back

The deliverable is **one tarball plus one short report message**. Both are described below.
`scripts/99_pack.sh` builds the tarball. You write the report.

## 1. The tarball

`99_pack.sh` writes `kvr_8gpu_results_<YYYYmmdd_HHMM>.tar.gz` next to the bundle directory.
It contains `out/`:

```
out/
  run.log                      stage begin/end lines
  summary.md                   human-readable tables (from scripts/summarize.py)
  summary.json                 the same numbers, machine-readable
  bundle_changes/              config.env used, sha256 of sources/scripts, operator.patch + NOTES.md (if any)
  files_written.log            restore files written to DATA_DIR
  00_preflight/                gpus.csv, nvidia-smi.txt, topo.txt, lscpu.txt, numa.txt, lsblk.txt,
                               mdstat.txt, nvme.txt, data_dir_mount.txt, summary.txt, failures.txt
  01_build/                    build.log, sha256.txt
  02_smoke/                    smoke.log, coord2.csv, logs/
  10_fio/                      read_{1M,512K,256K}_r{1,2}.json, read_under_write.json
  20_concurrent/               concurrent.csv, diskstats.csv, logs/, spec_*.json
  21_writeback/                writeback.csv, writer.json, diskstats.csv, logs/
  30_prefill/                  prefill.csv, prefill_*.log
  31_front_emul/               schedule.json, schedule.txt, front.csv, diskstats.csv, logs/
  32_arrivals/                 arrivals.csv, diskstats.csv
  40_e2e_front/                check_front.json, check_front.log, kvr_load.jsonl
  <stage>/status.json          {"stage", "status": ok|fail|skip, "note", start/end unix time}
```

Do not delete raw files to save space. The tarball should be well under 1 GB. If
`diskstats.csv` files make it larger, keep them anyway and say so. If the run was partial,
pack it anyway: partial results are useful.

Before handing it over, check that the tarball contains no secrets. Run:

```bash
tar -xzOf kvr_8gpu_results_*.tar.gz | grep -aiE "hf_[a-z0-9]{20,}|token|password" | head
```

It should print nothing relevant. Hostnames and GPU serial numbers are fine.

## 2. CSV schemas

### `concurrent.csv`, `writeback.csv`, `front.csv`, `arrivals.csv`

One row per restore process per launch (written by `scripts/launch.py`):

| column | meaning |
|---|---|
| `exp` | experiment: `concurrent_8b`, `concurrent_70b`, `writeback_*`, `front_*`, `arrivals_<geo>_x<0.25\|0.5\|1>` |
| `rep` | repetition index (0 … REPS-1); each process runs all repetitions, and every repetition starts on a common clock |
| `policy` | `fair` (no coordination) or `srpt` (shared-memory grant) |
| `G` | number of concurrent restore processes in this launch |
| `idx` | process index within the launch (0 … G-1) |
| `gpu` | physical GPU index |
| `geo` | `8b` (4 GiB) or `70b` (10 GiB) |
| `start_tokens` | first token loaded (k); 0 = full restore |
| `delay_s` | planned start offset after the common start T0 (front: emulated recompute time; arrivals: arrival time) |
| `bytes` | bytes read by this process |
| `start_rel_s` | restore start − T0 (CLOCK_MONOTONIC) |
| `end_rel_s` | restore end − T0 = **completion time** from the common start |
| `restore_s` | end − start for this process (includes waiting for the SRPT grant) |
| `mismatches` | words that failed verification; must be 0 |
| `exit` | process exit code if the process had already exited when the row was written, else empty; must be empty or 0 |

The completion time of a restore is `end_rel_s` for stages 20, 21 and 31. For stage 32 it
is `end_rel_s − delay_s`, measured from its own arrival.

### `prefill.csv`
`model, tokens, phase (warmup|run), iter, wall_s`: vLLM prefill time of a `tokens`-long
prompt with 1 output token.

### `schedule.json` (stage 31)
For each geometry:
- `G`: number of restores.
- `B_bytes_per_s`: measured single-restore bandwidth.
- `t_source`: whether t(n) was measured here, assumed, or a fallback.
- `k[i]`, `delay_s[i]`: the schedule.
- `pred`: model predictions of mean and max completion time for fair, srpt, and
  srpt+front, plus `recompute_full`.

## 3. Sanity checks before reporting

Check each of these. If one fails, report it as an observation; do not hide or "fix" the data.

1. **Stages 01 and 02**: every stage has `status.json`, and 01 and 02 are `ok`.
2. **Errors**: every `mismatches` is 0 and every `exit` is empty or 0. If not, count them.
3. **Single-restore bandwidth**: at G = 1, `bytes / restore_s` should be close to
   `min(fio 1M read, PCIe host-to-device bandwidth)`. On an H200 node with 8 PCIe-4 NVMe in
   RAID0, it was about 46–52 GB/s. A value far below 50% of fio means something else is
   limiting it; note what (NUMA, threads, QD, contention).
4. **Fair sharing**: at G > 1, `max end_rel_s` should grow about linearly with G, roughly
   `G × bytes / B_array`. Aggregate GB/s should stay near the array bandwidth.
5. **SRPT**:
   - Restores should finish staggered: sorted `end_rel_s` about 1×, 2×, …, G× of the
     single time.
   - The mean should be lower than under fair sharing.
   - The max should be about the same as fair.
   - Expected mean improvement at G = 8 is about 1.78× when the array is the bottleneck.
   - If srpt is *slower* than fair in the mean, report it with the numbers.
6. **Front (stage 31)**: compare the measured mean and max with `pred.srpt_front` and with
   stage 20's fair and srpt at the same G. Report the relative error of the prediction.
7. **Stage 40**: `pass_` true, and each check has `suffix_ok` and `equal_prefix_cache` true.
   Report `req2_overhead_s` per k.

## 4. The report message (paste back to the human)

Keep it short and factual. Use this structure:

```
# 8-GPU KV-restore results

Machine: <GPU model x count>, <array: drives, RAID level, filesystem>, CUDA <x>, driver <y>, kernel <z>
Run: <date>, <total minutes>, exclusive: yes/no (explain), operator changes: none / see NOTES.md

Stages: 00 ok, 01 ok, 02 ok, 10 ok/skip, 20 ok, 21 ..., 30 ..., 31 ..., 32 ..., 40 ...

Key numbers
- fio seq read 1M: X GB/s; under Y GB/s write: X' GB/s (p99 Z ms)
- single restore (G=1): 8b A s (B GB/s), 70b C s (D GB/s)
- G=8 fair  : 8b mean/max ..., 70b mean/max ...
- G=8 srpt  : 8b mean/max ... (Nx vs fair), 70b ...
- G=8 front : 8b mean/max ... vs predicted ..., t(n) source: measured/assumed/fallback
- write-back G=8: fair ..., srpt ...
- arrivals: srpt/fair mean response at x=0.25/0.5/1: ...
- t(n) 8b @4k/16k/32k: ...; 70b: ...
- e2e front (stage 40): PASS/FAIL, req2 overhead: ...

Sanity checks: <list any that failed, with the numbers>
Anomalies / notes: <anything unusual: other users, retries, failed launches, operator changes>

Attached: kvr_8gpu_results_<...>.tar.gz
```

Then attach or transfer the tarball as the human instructs. Do not upload it anywhere
yourself.

Copy the numbers from `out/summary.md`; do not recompute them by hand. If summary.md
looks wrong, say so and include the raw rows that show it.
