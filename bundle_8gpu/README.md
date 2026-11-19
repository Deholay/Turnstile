# 8-GPU shared-SSD KV-restore test bundle

Measures concurrent KV-cache restores from one local NVMe array on up to 8 GPUs, with and
without cross-GPU coordination.

**Operator (human or AI agent): read `running_guide.md` first, then `output_guide.md`.
Edit only `config.env`, then run the stages as described.**

Quick start (after reading the guides):

```bash
cp -r bundle_8gpu bundle_8gpu.orig
```

```bash
cd bundle_8gpu
```

```bash
$EDITOR config.env
```

With Slurm (after filling in account and partition, and after the human approves the SU):

```bash
sbatch job_1gpu.sbatch
```

```bash
sbatch job_8gpu.sbatch
```

Budget: about 200 SU in total at 60 SU per GPU-hour (see `running_guide.md` section 2a).

Deliverable: `kvr_8gpu_results_<timestamp>.tar.gz`, plus a short report in the format of
`output_guide.md` section 4.
