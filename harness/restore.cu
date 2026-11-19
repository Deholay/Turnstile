// restore.cu — standalone harness for the restore pipeline in kvr_core.cuh (P1).
//
// Paged KV here is one allocation, [layer][2][num_blocks][block_size][kv_heads][head_dim],
// with a random block table, like vLLM's FlashAttention layout.
// Correctness: the file holds a deterministic hash pattern; after each run a GPU kernel
// regenerates the expected value of every 8-byte word of paged KV and counts mismatches.
// With --start-tokens K, tokens < K must stay zero (the restore must not touch them).

#include "kvr_core.cuh"

#include <random>
#include <string>

struct Cfg {
  int layers = 80, kv_heads = 8, head_dim = 128, elem = 2;
  int tokens = 32768, seg_tokens = 128, block_size = 16;
  int start_tokens = 0;  // tokens already on the GPU: restore only [start, tokens)
  int threads = 8, qd = 8, ring = 256, batch = 32;
  int runs = 5, warmup = 3;
  bool copy_mode = true, no_overlap = false, write = false, verify = true;
  uint64_t seed = 1;
  std::string file, csv, label, trace, go_file, coord;
};

// ---------------------------------------------------------------- pattern
__host__ __device__ static inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
// value of 8-byte word wi of token `tok`, side (0=K,1=V), layer
__host__ __device__ static inline uint64_t pat(uint64_t seed, int layer, int side, int tok,
                                               int wi, int tokens, int W) {
  uint64_t key = (((uint64_t)layer * 2 + side) * tokens + tok) * W + wi;
  return mix64(key ^ (seed * 0xd1b54a32d192ed03ULL));
}

__global__ void verify_kernel(const uint64_t* __restrict__ kv, const int* __restrict__ inv,
                              int layers, int nb, int block_size, int W, int tokens, int tok0,
                              uint64_t seed, unsigned long long* bad) {
  long words_per_block = (long)block_size * W;
  long total = (long)layers * 2 * nb * words_per_block;
  unsigned long long local = 0;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    long blk = i / words_per_block;
    long r = i - blk * words_per_block;
    int off = (int)(r / W), wi = (int)(r % W);
    int phys = (int)(blk % nb);
    long ls = blk / nb;
    int side = (int)(ls % 2), layer = (int)(ls / 2);
    int tok = inv[phys] * block_size + off;
    uint64_t want = tok < tok0 ? 0 : pat(seed, layer, side, tok, wi, tokens, W);
    if (kv[i] != want) local++;
  }
  if (local) atomicAdd(bad, local);
}

// ---------------------------------------------------------------- write mode
static void write_file(const Cfg& c, const Geo& g) {
  int fd = open(c.file.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
  if (fd < 0) die("open for write");
  if (ftruncate(fd, (off_t)g.file_bytes) != 0) die("ftruncate");
  std::atomic<long> next{0};
  int W = (int)(g.tok_bytes / 8);
  double t0 = now_s();
  auto worker = [&]() {
    void* buf;
    if (posix_memalign(&buf, 4096, g.seg_bytes)) die("posix_memalign");
    uint64_t* w = (uint64_t*)buf;
    for (long j; (j = next.fetch_add(1)) < g.total;) {
      int layer = (int)(j / g.S), seg = (int)(j % g.S);
      size_t k = 0;
      for (int side = 0; side < 2; side++)
        for (int t = 0; t < c.seg_tokens; t++)
          for (int wi = 0; wi < W; wi++)
            w[k++] = pat(c.seed, layer, side, seg * c.seg_tokens + t, wi, c.tokens, W);
      if (pwrite(fd, buf, g.seg_bytes, (off_t)(j * g.seg_bytes)) != (ssize_t)g.seg_bytes)
        die("pwrite");
    }
    free(buf);
  };
  int nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> ts;
  for (int i = 0; i < nt; i++) ts.emplace_back(worker);
  for (auto& t : ts) t.join();
  fsync(fd);
  close(fd);
  double dt = now_s() - t0;
  printf("wrote %s: %.2f GiB in %.2f s (%.2f GB/s)\n", c.file.c_str(), g.file_bytes / 1073741824.0,
         dt, g.file_bytes / dt / 1e9);
}

// ---------------------------------------------------------------- main
static void usage() {
  fprintf(stderr,
          "restore --file PATH [--write] [--layers 80 --kv-heads 8 --head-dim 128 --tokens 32768]\n"
          "        [--seg-tokens 128 --block-size 16] [--start-tokens 0] [--threads 8 --qd 8 --ring 256 --batch 32]\n"
          "        [--mode copy|direct] [--no-overlap] [--no-verify] [--runs 5 --warmup 3]\n"
          "        [--csv out.csv --label name] [--seed 1] [--trace io.csv]\n"
          "        [--go-file F: after setup, wait until F exists (start several GPUs together)]\n"
          "        [--coord NAME: SRPT grant through shared memory /NAME, shared with other\n"
          "         restore processes; prints monotonic start/end of each run]\n");
  exit(2);
}

int main(int argc, char** argv) {
  Cfg c;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
    if (a == "--file") c.file = nx();
    else if (a == "--write") c.write = true;
    else if (a == "--layers") c.layers = atoi(nx());
    else if (a == "--kv-heads") c.kv_heads = atoi(nx());
    else if (a == "--head-dim") c.head_dim = atoi(nx());
    else if (a == "--tokens") c.tokens = atoi(nx());
    else if (a == "--seg-tokens") c.seg_tokens = atoi(nx());
    else if (a == "--block-size") c.block_size = atoi(nx());
    else if (a == "--start-tokens") c.start_tokens = atoi(nx());
    else if (a == "--threads") c.threads = atoi(nx());
    else if (a == "--qd") c.qd = atoi(nx());
    else if (a == "--ring") c.ring = atoi(nx());
    else if (a == "--batch") c.batch = atoi(nx());
    else if (a == "--mode") { std::string m = nx(); if (m != "copy" && m != "direct") usage(); c.copy_mode = (m == "copy"); }
    else if (a == "--no-overlap") c.no_overlap = true;
    else if (a == "--no-verify") c.verify = false;
    else if (a == "--runs") c.runs = atoi(nx());
    else if (a == "--warmup") c.warmup = atoi(nx());
    else if (a == "--csv") c.csv = nx();
    else if (a == "--label") c.label = nx();
    else if (a == "--trace") c.trace = nx();
    else if (a == "--go-file") c.go_file = nx();
    else if (a == "--coord") c.coord = nx();
    else if (a == "--seed") c.seed = strtoull(nx(), nullptr, 10);
    else usage();
  }
  if (c.file.empty()) usage();
  Geo g = make_geo(c.layers, c.tokens, c.seg_tokens, c.block_size,
                   (size_t)c.kv_heads * c.head_dim * c.elem, c.start_tokens);
  if (!geo_ok(g, c.tokens)) {
    fprintf(stderr, "bad geometry: tokens%%seg, seg%%block, seg_bytes%%4096, tok_bytes%%16, "
                    "start%%block must be 0 and start < tokens\n");
    return 2;
  }
  int nb = c.tokens / c.block_size;
  size_t kv_bytes = (size_t)c.layers * 2 * nb * g.block_bytes;
  if (c.no_overlap) c.ring = (int)g.n;
  printf("geometry: %d layers x %d tokens, tok_bytes=%zu, item=%zu KiB, items=%ld, file=%.2f GiB\n",
         c.layers, c.tokens, g.tok_bytes, g.seg_bytes >> 10, g.total, g.file_bytes / 1073741824.0);
  if (!c.write && c.start_tokens)
    printf("partial: start_tokens=%d -> segments [%d, %d) per layer, items=%ld, read=%.2f GiB\n",
           c.start_tokens, g.s0, g.S, g.n, g.read_bytes / 1073741824.0);

  if (c.write) { write_file(c, g); return 0; }

  int fd = open(c.file.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) die("open");
  off_t fsz = lseek(fd, 0, SEEK_END);
  if (fsz < (off_t)g.file_bytes) { fprintf(stderr, "file too small: %ld < %zu\n", (long)fsz, g.file_bytes); return 1; }

  cudaDeviceProp prop;
  int dev;
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("gpu: %s, mode=%s threads=%d qd=%d ring=%d batch=%d overlap=%s\n", prop.name,
         c.copy_mode ? "copy" : "direct", c.threads, c.qd, c.ring, c.batch,
         c.no_overlap ? "no" : "yes");

  Pipeline P;
  P.threads = c.threads;
  P.qd = c.qd;
  P.batch = c.batch;
  P.copy_mode = c.copy_mode;
  P.no_overlap = c.no_overlap;
  pipeline_init(P, (size_t)c.ring * g.seg_bytes);
  if (!c.coord.empty()) P.coord = coord_open(c.coord.c_str());

  char* kv;
  CK(cudaMalloc(&kv, kv_bytes));
  std::vector<char*> lp(c.layers);
  for (int l = 0; l < c.layers; l++) lp[l] = kv + (size_t)l * 2 * nb * g.block_bytes;
  char** lp_dev;
  CK(cudaMalloc(&lp_dev, c.layers * sizeof(char*)));
  CK(cudaMemcpy(lp_dev, lp.data(), c.layers * sizeof(char*), cudaMemcpyHostToDevice));
  long side_stride = (long)nb * g.block_bytes;

  std::vector<int> bt(nb), inv(nb);
  std::iota(bt.begin(), bt.end(), 0);
  std::shuffle(bt.begin(), bt.end(), std::mt19937_64(c.seed));
  for (int i = 0; i < nb; i++) inv[bt[i]] = i;
  int *bt_dev, *inv_dev;
  CK(cudaMalloc(&bt_dev, nb * sizeof(int)));
  CK(cudaMalloc(&inv_dev, nb * sizeof(int)));
  CK(cudaMemcpy(bt_dev, bt.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(inv_dev, inv.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
  unsigned long long* bad_dev;
  CK(cudaMalloc(&bad_dev, sizeof(unsigned long long)));

  FILE* csv = nullptr;
  if (!c.csv.empty()) {
    bool fresh = access(c.csv.c_str(), F_OK) != 0;
    csv = fopen(c.csv.c_str(), "a");
    if (!csv) die("csv");
    if (fresh)
      fprintf(csv, "label,gpu,mode,overlap,layers,kv_heads,head_dim,tokens,seg_kib,threads,qd,ring,batch,"
                   "run,total_s,read_s,gbps,layer0_ms,layer_last_ms,mismatches\n");
  }
  FILE* trf = nullptr;
  std::vector<IoRec> trace;
  if (!c.trace.empty()) {
    bool fresh = access(c.trace.c_str(), F_OK) != 0;
    trf = fopen(c.trace.c_str(), "a");
    if (!trf) die("trace");
    if (fresh) fputs(trace_header(), trf);
    trace.resize(g.n);
    P.trace = trace.data();
  }
  std::vector<double> totals;
  if (!c.go_file.empty())
    while (access(c.go_file.c_str(), F_OK) != 0) usleep(1000);
  for (int r = 0; r < c.warmup + c.runs; r++) {
    CK(cudaMemset(kv, 0, kv_bytes));
    CK(cudaDeviceSynchronize());
    if (P.coord) coord_register(P.coord, (long long)g.read_bytes);
#ifdef KVR_NVTX
    nvtxRangePushA(r < c.warmup ? "warmup" : "run");
#endif
    RunResult res = pipeline_restore(P, g, fd, lp_dev, side_stride,
                                     bt_dev + c.start_tokens / c.block_size);
#ifdef KVR_NVTX
    nvtxRangePop();
#endif
    if (P.coord) {
      coord_done(P.coord);
      printf("coord pid %d bytes %zu start %.6f end %.6f\n", getpid(), g.read_bytes, res.t0,
             res.t0 + res.total_s);
    }
    if (res.total_s < 0) { fprintf(stderr, "run failed\n"); return 1; }
    if (c.verify) {
      CK(cudaMemset(bad_dev, 0, sizeof(unsigned long long)));
      verify_kernel<<<1024, 256>>>((const uint64_t*)kv, inv_dev, c.layers, nb, c.block_size,
                                   (int)(g.tok_bytes / 8), c.tokens, c.start_tokens, c.seed,
                                   bad_dev);
      CK(cudaMemcpy(&res.bad, bad_dev, sizeof res.bad, cudaMemcpyDeviceToHost));
    }
    bool warm = r < c.warmup;
    double gbps = g.read_bytes / res.total_s / 1e9;
    printf("%s %d: total %.4f s  read %.4f s  %.2f GB/s  layer0 %.2f ms  last %.2f ms  mismatches %llu\n",
           warm ? "warmup" : "run", warm ? r : r - c.warmup, res.total_s, res.read_s, gbps,
           res.layer0_ms, res.layer_last_ms, res.bad);
    if (res.bad) { fprintf(stderr, "VERIFY FAILED\n"); return 1; }
    if (!warm) {
      totals.push_back(res.total_s);
      if (trf) trace_write(trf, c.label.c_str(), r - c.warmup, trace.data(), g, res.t0);
      if (csv)
        fprintf(csv, "%s,%s,%s,%s,%d,%d,%d,%d,%zu,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f,%.3f,%.3f,%llu\n",
                c.label.c_str(), prop.name, c.copy_mode ? "copy" : "direct",
                c.no_overlap ? "no" : "yes", c.layers, c.kv_heads, c.head_dim, c.tokens,
                g.seg_bytes >> 10, c.threads, c.qd, c.ring, c.batch, r - c.warmup, res.total_s,
                res.read_s, gbps, res.layer0_ms, res.layer_last_ms, res.bad);
    }
  }
  if (csv) fclose(csv);
  if (trf) fclose(trf);
  if (!totals.empty()) {
    std::sort(totals.begin(), totals.end());
    size_t n = totals.size();
    printf("median %.4f s (%.2f GB/s), IQR [%.4f, %.4f]\n", totals[n / 2],
           g.read_bytes / totals[n / 2] / 1e9, totals[n / 4], totals[(3 * n) / 4]);
  }
  pipeline_free(P);
  close(fd);
  return 0;
}
