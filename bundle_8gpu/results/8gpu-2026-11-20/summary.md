# 8-GPU restore test summary

## Stages

- 00_preflight: **ok** 
- 01_build: **ok** 
- 02_smoke: **ok** 
- 10_fio: **skip** fio not installed
- 20_concurrent: **ok** 
- 21_writeback: **skip** fio not installed
- 30_prefill: **ok** 8b ok; 70b ok
- 31_front_emul: **ok** 
- 32_arrivals: **ok** 

## Machine

- 0, NVIDIA H200, 143771 MiB, 550.127.08, 00000000:18:00.0, 9.0
- 1, NVIDIA H200, 143771 MiB, 550.127.08, 00000000:3A:00.0, 9.0
- 2, NVIDIA H200, 143771 MiB, 550.127.08, 00000000:4B:00.0, 9.0
- 3, NVIDIA H200, 143771 MiB, 550.127.08, 00000000:5C:00.0, 9.0
- 4, NVIDIA H200, 143771 MiB, 550.127.08, 00000000:9A:00.0, 9.0
- 5, NVIDIA H200, 143771 MiB, 550.127.08, 00000000:BA:00.0, 9.0
- 6, NVIDIA H200, 143771 MiB, 550.127.08, 00000000:CA:00.0, 9.0
- 7, NVIDIA H200, 143771 MiB, 550.127.08, 00000000:DA:00.0, 9.0
- data_dir_fstype: xfs
- data_dir_free_gib: 28414
- kernel: 4.18.0-513.24.1.el8_9.x86_64

## Concurrent restores (stage 20)

| exp | G | policy | reps | mean s | max s | agg GB/s | srpt/fair |
|---|---|---|---|---|---|---|---|
| concurrent_70b | 1 | fair | 5 | 0.235 | 0.235 | 45.682 | - |
| concurrent_70b | 2 | fair | 5 | 0.411 | 0.417 | 51.554 | - |
| concurrent_70b | 2 | srpt | 5 | 0.316 | 0.421 | 51.017 | 1.301 |
| concurrent_70b | 4 | fair | 5 | 0.816 | 0.819 | 52.454 | - |
| concurrent_70b | 4 | srpt | 5 | 0.524 | 0.832 | 51.605 | 1.555 |
| concurrent_70b | 8 | fair | 5 | 1.600 | 1.702 | 50.479 | - |
| concurrent_70b | 8 | srpt | 5 | 0.991 | 1.807 | 47.535 | 1.614 |
| concurrent_8b | 1 | fair | 5 | 0.093 | 0.093 | 46.055 | - |
| concurrent_8b | 2 | fair | 5 | 0.171 | 0.171 | 50.232 | - |
| concurrent_8b | 2 | srpt | 5 | 0.131 | 0.173 | 49.694 | 1.300 |
| concurrent_8b | 4 | fair | 5 | 0.330 | 0.334 | 51.488 | - |
| concurrent_8b | 4 | srpt | 5 | 0.214 | 0.339 | 50.700 | 1.539 |
| concurrent_8b | 8 | fair | 5 | 0.650 | 0.688 | 49.930 | - |
| concurrent_8b | 8 | srpt | 5 | 0.405 | 0.726 | 47.312 | 1.605 |

## SRPT + recompute-front, emulated (stage 31)

| exp | G | policy | reps | mean s | max s | agg GB/s | srpt/fair |
|---|---|---|---|---|---|---|---|
| front_70b | 8 | srpt | 5 | 0.929 | 1.599 | 45.422 | - |
| front_8b | 8 | srpt | 5 | 0.308 | 0.488 | 45.662 | - |

## Poisson arrivals, response time from arrival (stage 32)

| exp | G | policy | reps | mean s | max s | agg GB/s | srpt/fair |
|---|---|---|---|---|---|---|---|
| arrivals_70b_x0.25 | 8 | fair | 5 | 1.389 | 1.535 | 50.006 | - |
| arrivals_70b_x0.25 | 8 | srpt | 5 | 0.869 | 1.505 | 47.529 | 1.599 |
| arrivals_70b_x0.5 | 8 | fair | 5 | 0.763 | 0.892 | 49.927 | - |
| arrivals_70b_x0.5 | 8 | srpt | 5 | 0.536 | 0.880 | 47.274 | 1.424 |
| arrivals_70b_x1 | 8 | fair | 5 | 0.555 | 0.780 | 47.831 | - |
| arrivals_70b_x1 | 8 | srpt | 5 | 0.411 | 0.631 | 46.409 | 1.352 |
| arrivals_8b_x0.25 | 8 | fair | 5 | 0.553 | 0.611 | 50.171 | - |
| arrivals_8b_x0.25 | 8 | srpt | 5 | 0.348 | 0.602 | 47.594 | 1.588 |
| arrivals_8b_x0.5 | 8 | fair | 5 | 0.314 | 0.368 | 49.825 | - |
| arrivals_8b_x0.5 | 8 | srpt | 5 | 0.220 | 0.358 | 47.445 | 1.428 |
| arrivals_8b_x1 | 8 | fair | 5 | 0.223 | 0.316 | 47.757 | - |
| arrivals_8b_x1 | 8 | srpt | 5 | 0.166 | 0.257 | 46.073 | 1.342 |

## Front schedule and model prediction

- 8b: B=46.3 GB/s, t(n) measured, k=[0, 5504, 8960, 11776, 13952, 15872, 17408, 18816]; predicted mean fair 0.742 / srpt 0.418 / srpt+front 0.308 s
- 70b: B=45.8 GB/s, t(n) measured, k=[0, 1792, 3328, 4736, 6016, 7168, 8192, 9216]; predicted mean fair 1.874 / srpt 1.054 / srpt+front 0.940 s

## Recompute t(n) (s, median)

- 8b: 1024: 0.018, 2048: 0.033, 4096: 0.066, 6144: 0.105, 8192: 0.150, 12288: 0.249, 16384: 0.363, 20480: 0.490, 24576: 0.632, 28672: 0.785, 32768: 0.957
- 70b: 1024: 0.133, 2048: 0.264, 4096: 0.550, 6144: 0.873, 8192: 1.221, 12288: 1.932, 16384: 2.894, 20480: 3.785, 24576: 4.742, 28672: 5.848, 32768: 7.015
