// kvr_core.cuh — restore pipeline shared by the standalone harness (restore.cu) and the
// vLLM connector library (kvr_lib.cu).
//
// SSD layout: one file per stored prefix, layer-major. Item j = (layer j / S, segment j % S),
// S = tokens / seg_tokens; each item is [K: seg_tokens x tok_bytes][V: seg_tokens x tok_bytes].
//
// Partial restore (recompute-front): with start_tok > 0 the caller already has the KV of
// tokens [0, start_tok) on the GPU (prefix-cache hit or recompute). Only segments
// [start_tok / seg_tokens, S) of every layer are read, tokens < start_tok inside the first of
// them are not written, and the block table covers tokens [start_tok, tokens) only.
// Restore index i in [0, n) maps to file item item_of(i); ring slots, ordering and the I/O
// trace use i.
//
// Pipeline:
//   S1  N reader threads, each keeps QD O_DIRECT reads in flight (raw Linux AIO syscalls).
//   S2  pinned ring of R item slots; slot j%R may be reused once item j-R is on the GPU.
//   S3/S4  the calling thread batches ready items and launches a scatter kernel that writes
//       paged KV: layer_ptrs[layer] + side*side_stride + block*block_bytes + off*tok_bytes.
//         copy:   cudaMemcpyAsync slot -> device staging ring, then device->device scatter
//         direct: kernel reads the mapped pinned slot over PCIe
//   Per layer: a CUDA event after the batch that completes the layer.
// Optional I/O trace (Pipeline::trace): submit/complete time of every read, CLOCK_MONOTONIC
// seconds, so it lines up with a /proc/diskstats sampler using Python's time.monotonic().
#pragma once
#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "kvr_coord.h"

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__,               \
              cudaGetErrorString(e_));                                                \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

static void die(const char* m) { perror(m); exit(1); }

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ---------------------------------------------------------------- geometry
struct Geo {
  int layers, seg_tokens, block_size, S;
  long total;
  size_t tok_bytes, seg_bytes, block_bytes, file_bytes;
  int tok0, s0, Sr;  // first token written, first segment read, segments read per layer
  long n;            // items read = layers * Sr
  size_t read_bytes;
  long item_of(long i) const { return (i / Sr) * S + s0 + i % Sr; }
};

static Geo make_geo(int layers, int tokens, int seg_tokens, int block_size, size_t tok_bytes,
                    int start_tok = 0) {
  Geo g;
  g.layers = layers;
  g.seg_tokens = seg_tokens;
  g.block_size = block_size;
  g.tok_bytes = tok_bytes;
  g.seg_bytes = 2 * (size_t)seg_tokens * tok_bytes;
  g.block_bytes = (size_t)block_size * tok_bytes;
  g.S = tokens / seg_tokens;
  g.total = (long)layers * g.S;
  g.file_bytes = (size_t)g.total * g.seg_bytes;
  g.tok0 = start_tok;
  g.s0 = start_tok / seg_tokens;
  g.Sr = g.S - g.s0;
  g.n = (long)layers * g.Sr;
  g.read_bytes = (size_t)g.n * g.seg_bytes;
  return g;
}

static bool geo_ok(const Geo& g, int tokens) {
  return tokens % g.seg_tokens == 0 && g.seg_tokens % g.block_size == 0 &&
         g.seg_bytes % 4096 == 0 && g.tok_bytes % 16 == 0 && g.tok0 >= 0 &&
         g.tok0 < tokens && g.tok0 % g.block_size == 0;
}

// ---------------------------------------------------------------- kernel
constexpr int MAXB = 64;
struct Batch {
  int n;
  int item[MAXB];
  const char* src[MAXB];
};

struct ScatterGeo {
  int S, seg_tokens, block_size, tok0;
  long tok_bytes, seg_bytes, block_bytes, side_stride;
};

__global__ void scatter_kernel(Batch b, ScatterGeo g, const int* __restrict__ bt,
                               char* const* __restrict__ layer_ptrs) {
  int bi = blockIdx.y;
  int item = b.item[bi];
  const uint4* src = reinterpret_cast<const uint4*>(b.src[bi]);
  int layer = item / g.S, seg = item % g.S;
  char* base = layer_ptrs[layer];
  long side_bytes = g.seg_bytes / 2;
  long n16 = g.seg_bytes / 16;
  for (long v = blockIdx.x * (long)blockDim.x + threadIdx.x; v < n16;
       v += (long)gridDim.x * blockDim.x) {
    long o = v * 16;
    int side = (int)(o / side_bytes);
    long rem = o - side * side_bytes;
    int t = (int)(rem / g.tok_bytes);
    long within = rem - (long)t * g.tok_bytes;
    int tok = seg * g.seg_tokens + t;
    if (tok < g.tok0) continue;  // already on the GPU
    int phys = bt[(tok - g.tok0) / g.block_size];
    int off = tok % g.block_size;
    char* dst = base + side * g.side_stride + (long)phys * g.block_bytes +
                (long)off * g.tok_bytes + within;
    *reinterpret_cast<uint4*>(dst) = src[v];
  }
}

// ---------------------------------------------------------------- aio
static long io_setup_(unsigned n, aio_context_t* c) { return syscall(SYS_io_setup, n, c); }
static long io_destroy_(aio_context_t c) { return syscall(SYS_io_destroy, c); }
static long io_submit_(aio_context_t c, long n, iocb** p) { return syscall(SYS_io_submit, c, n, p); }
static long io_getevents_(aio_context_t c, long mn, long mx, io_event* e, timespec* t) {
  return syscall(SYS_io_getevents, c, mn, mx, e, t);
}

// ---------------------------------------------------------------- pipeline
struct IoRec {
  double sub, done;  // monotonic seconds
  int thread;
};

struct Pipeline {
  int threads = 8, qd = 16, batch = 32;
  IoRec* trace = nullptr;  // caller-owned, >= items per restore; nullptr = off
  Coord* coord = nullptr;  // SRPT gate shared with other processes; caller registers; nullptr = off
  bool copy_mode = true, no_overlap = false;
  size_t ring_bytes = 0;
  char *ring_host = nullptr, *ring_dev_mapped = nullptr, *ring_dev = nullptr;
  cudaStream_t cs, xs;  // compute (scatter), copy
  std::vector<cudaEvent_t> pool, layer_ev;
  cudaEvent_t start_ev, copy_ev;
};

static void pipeline_init(Pipeline& P, size_t ring_bytes) {
  P.ring_bytes = ring_bytes;
  CK(cudaHostAlloc((void**)&P.ring_host, ring_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
  memset(P.ring_host, 0, ring_bytes);  // first touch on the bound NUMA node
  CK(cudaHostGetDevicePointer((void**)&P.ring_dev_mapped, P.ring_host, 0));
  if (P.copy_mode) CK(cudaMalloc(&P.ring_dev, ring_bytes));
  CK(cudaStreamCreateWithFlags(&P.cs, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&P.xs, cudaStreamNonBlocking));
  CK(cudaEventCreate(&P.start_ev));
  CK(cudaEventCreateWithFlags(&P.copy_ev, cudaEventDisableTiming));
}

static void pipeline_free(Pipeline& P) {
  for (auto e : P.pool) cudaEventDestroy(e);
  for (auto e : P.layer_ev) cudaEventDestroy(e);
  cudaEventDestroy(P.start_ev);
  cudaEventDestroy(P.copy_ev);
  cudaStreamDestroy(P.cs);
  cudaStreamDestroy(P.xs);
  if (P.ring_dev) cudaFree(P.ring_dev);
  cudaFreeHost(P.ring_host);
}

struct Shared {
  std::atomic<long> next{0};
  std::atomic<long> consumed{0};  // items whose scatter has completed on GPU
  std::atomic<long>* slot_item;   // slot -> (item+1) once its read completed
  std::atomic<int> armed{0};
  std::atomic<bool> go{false};
  std::atomic<bool> fail{false};
};

struct RunResult {
  double t0, total_s, read_s;  // t0: monotonic start, for the I/O trace
  float layer0_ms, layer_last_ms;
  unsigned long long bad;
};

static void reader(int tid, int qd, long R, const Geo& g, Shared& sh, int fd, char* ring_host,
                   IoRec* trace, Coord* coord, double* t_last) {
  aio_context_t ctx = 0;
  if (io_setup_(qd, &ctx) != 0) die("io_setup");
  std::vector<iocb> cbs(qd);
  std::vector<int> free_cb(qd);
  std::iota(free_cb.begin(), free_cb.end(), 0);
  std::vector<io_event> evs(qd);
  int inflight = 0;
  bool fetched_all = false;
  timespec zero{0, 0};
  double last = 0;

  auto reap = [&](long min) {
    long n = io_getevents_(ctx, min, qd, evs.data(), min ? nullptr : &zero);
    if (n < 0) {
      if (errno == EINTR) return;
      die("io_getevents");
    }
    double t = n > 0 ? now_s() : 0;
    for (long i = 0; i < n; i++) {
      long j = (long)evs[i].data;
      if (trace) trace[j].done = t;
      if (evs[i].res != (long long)g.seg_bytes) {
        fprintf(stderr, "read item %ld: res=%lld\n", j, (long long)evs[i].res);
        sh.fail = true;
      }
      sh.slot_item[j % R].store(j + 1, std::memory_order_release);
      free_cb.push_back((int)(((iocb*)evs[i].obj) - cbs.data()));
      inflight--;
    }
    if (n > 0) last = t;
  };

  sh.armed.fetch_add(1);
  while (!sh.go.load(std::memory_order_acquire)) {}

  while (!sh.fail) {
    while (inflight < qd && !fetched_all && !sh.fail) {
      if (coord && !coord_turn(coord)) {  // another restore has fewer bytes left
        if (inflight) break;              // reap below, then ask again
        usleep(50);
        continue;
      }
      long j = sh.next.fetch_add(1);
      if (j >= g.n) { fetched_all = true; break; }
      if (coord) coord_consume(coord, (long long)g.seg_bytes);
      while (j >= sh.consumed.load(std::memory_order_acquire) + R) {
        if (inflight) reap(0);
        else std::this_thread::yield();
      }
      int k = free_cb.back();
      free_cb.pop_back();
      iocb& cb = cbs[k];
      memset(&cb, 0, sizeof cb);
      cb.aio_lio_opcode = IOCB_CMD_PREAD;
      cb.aio_fildes = fd;
      cb.aio_buf = (uint64_t)(ring_host + (j % R) * g.seg_bytes);
      cb.aio_nbytes = g.seg_bytes;
      cb.aio_offset = (int64_t)(g.item_of(j) * g.seg_bytes);
      cb.aio_data = (uint64_t)j;
      iocb* p = &cb;
      if (trace) trace[j] = IoRec{now_s(), 0, tid};
      if (io_submit_(ctx, 1, &p) != 1) die("io_submit");
      inflight++;
    }
    if ((fetched_all || sh.fail) && inflight == 0) break;
    if (inflight) reap(1);
  }
  // drain anything still in flight after a failure so buffers are not written later
  while (inflight > 0) reap(1);
  io_destroy_(ctx);
  *t_last = last;
}

// Restore the g.n items of segments [g.s0, S) of every layer from fd into paged KV; bt_dev
// covers tokens [g.tok0, tokens). layer_ptrs_dev / bt_dev are device arrays.
// Returns with all GPU work complete. On a read error returns total_s < 0.
static RunResult pipeline_restore(Pipeline& P, const Geo& g, int fd, char* const* layer_ptrs_dev,
                                  long side_stride, const int* bt_dev) {
  long R = (long)(P.ring_bytes / g.seg_bytes);
  if (P.no_overlap && R < g.n) { fprintf(stderr, "no_overlap needs ring >= file\n"); exit(1); }
  if (R < 1) { fprintf(stderr, "ring smaller than one item\n"); exit(1); }
  while ((long)P.pool.size() < R + 1) {
    cudaEvent_t e;
    CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    P.pool.push_back(e);
  }
  while ((int)P.layer_ev.size() < g.layers) {
    cudaEvent_t e;
    CK(cudaEventCreate(&e));
    P.layer_ev.push_back(e);
  }

  Shared sh;
  std::vector<std::atomic<long>> slot_item(R);
  for (auto& s : slot_item) s.store(0);
  sh.slot_item = slot_item.data();

  std::vector<double> t_last(P.threads, 0);
  std::vector<std::thread> ts;
  for (int i = 0; i < P.threads; i++)
    ts.emplace_back(reader, i, P.qd, R, std::cref(g), std::ref(sh), fd, P.ring_host, P.trace,
                    P.coord, &t_last[i]);
  while (sh.armed.load() < P.threads) {}

  ScatterGeo sg{g.S, g.seg_tokens, g.block_size, g.tok0, (long)g.tok_bytes, (long)g.seg_bytes,
                (long)g.block_bytes, side_stride};
  const int threads_per_block = 256, blocks_per_item = 32;

  long Pn = R + 1;  // events in use
  std::vector<long> q_upto(Pn);
  long q_head = 0, q_tail = 0;
  auto poll = [&](bool block) {
    while (q_head < q_tail) {
      cudaEvent_t e = P.pool[q_head % Pn];
      if (block) CK(cudaEventSynchronize(e));
      else {
        cudaError_t r = cudaEventQuery(e);
        if (r == cudaErrorNotReady) return;
        CK(r);
      }
      sh.consumed.store(q_upto[q_head % Pn], std::memory_order_release);
      q_head++;
    }
  };

  double t0 = now_s();
  CK(cudaEventRecord(P.start_ev, P.cs));
  sh.go.store(true, std::memory_order_release);

  if (P.no_overlap)
    for (long i = 0; i < g.n; i++)
      while (slot_item[i % R].load(std::memory_order_acquire) != i + 1 && !sh.fail) {}

  long j = 0;
  while (j < g.n && !sh.fail) {
    while (slot_item[j % R].load(std::memory_order_acquire) != j + 1) {
      poll(false);
      if (sh.fail) break;
    }
    if (sh.fail) break;
    Batch b;
    b.n = 0;
    long lim = std::min<long>({(long)P.batch, (long)MAXB, g.n - j});
    while (b.n < lim &&
           slot_item[(j + b.n) % R].load(std::memory_order_acquire) == j + b.n + 1) {
      long it = j + b.n;
      b.item[b.n] = (int)g.item_of(it);
      b.src[b.n] = (P.copy_mode ? P.ring_dev : P.ring_dev_mapped) + (it % R) * g.seg_bytes;
      b.n++;
    }
    if (P.copy_mode) {
      long s0 = j % R, n = b.n;
      long first = std::min(n, R - s0);
      CK(cudaMemcpyAsync(P.ring_dev + s0 * g.seg_bytes, P.ring_host + s0 * g.seg_bytes,
                         first * g.seg_bytes, cudaMemcpyHostToDevice, P.xs));
      if (n > first)
        CK(cudaMemcpyAsync(P.ring_dev, P.ring_host, (n - first) * g.seg_bytes,
                           cudaMemcpyHostToDevice, P.xs));
      CK(cudaEventRecord(P.copy_ev, P.xs));
      CK(cudaStreamWaitEvent(P.cs, P.copy_ev, 0));
    }
    scatter_kernel<<<dim3(blocks_per_item, b.n), threads_per_block, 0, P.cs>>>(b, sg, bt_dev,
                                                                              layer_ptrs_dev);
    CK(cudaGetLastError());
    long end = j + b.n;
    for (long l = j / g.Sr; l < g.layers && (l + 1) * (long)g.Sr <= end; l++)
      if ((l + 1) * (long)g.Sr > j) CK(cudaEventRecord(P.layer_ev[l], P.cs));
    if (q_tail - q_head >= Pn) poll(true);
    CK(cudaEventRecord(P.pool[q_tail % Pn], P.cs));
    q_upto[q_tail % Pn] = end;
    q_tail++;
    j = end;
    poll(false);
  }
  if (sh.fail) sh.consumed.store(g.n + R);  // release readers waiting for ring space
  poll(true);
  CK(cudaStreamSynchronize(P.cs));
  double t1 = now_s();
  for (auto& t : ts) t.join();

  RunResult r{};
  if (sh.fail) { r.total_s = -1; return r; }
  r.t0 = t0;
  r.total_s = t1 - t0;
  r.read_s = *std::max_element(t_last.begin(), t_last.end()) - t0;
  CK(cudaEventElapsedTime(&r.layer0_ms, P.start_ev, P.layer_ev[0]));
  CK(cudaEventElapsedTime(&r.layer_last_ms, P.start_ev, P.layer_ev[g.layers - 1]));
  return r;
}

// One CSV row per read of the last restore. Header: trace_header().
static const char* trace_header() {
  return "tag,run,item,layer,seg,offset,bytes,thread,submit_s,complete_s,t0_mono\n";
}
static void trace_write(FILE* f, const char* tag, int run, const IoRec* tr, const Geo& g,
                        double t0) {
  for (long j = 0; j < g.n; j++) {
    long it = g.item_of(j);
    fprintf(f, "%s,%d,%ld,%ld,%ld,%ld,%zu,%d,%.7f,%.7f,%.6f\n", tag, run, it, it / g.S, it % g.S,
            (long)(it * g.seg_bytes), g.seg_bytes, tr[j].thread, tr[j].sub - t0, tr[j].done - t0,
            t0);
  }
}
