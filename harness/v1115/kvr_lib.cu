// kvr_lib.cu — C API around the restore pipeline, loaded by the vLLM connector via ctypes.
// One handle per process (per GPU). kvr_load is synchronous: when it returns, the KV of the
// first `tokens` tokens of every layer is in vLLM's paged KV cache.
#include "kvr_core.cuh"

namespace {
struct Handle {
  Pipeline P;
  int device;
  char** lp_dev = nullptr;
  int lp_cap = 0;
  int* bt_dev = nullptr;
  int bt_cap = 0;
};
}  // namespace

extern "C" {

void* kvr_open(int device, int threads, int qd, int batch, unsigned long long ring_bytes,
               int copy_mode) {
  CK(cudaSetDevice(device));
  Handle* h = new Handle;
  h->device = device;
  h->P.threads = threads;
  h->P.qd = qd;
  h->P.batch = batch;
  h->P.copy_mode = copy_mode != 0;
  pipeline_init(h->P, ring_bytes);
  return h;
}

// layer_ptrs: host array of `layers` device pointers (start of each layer's KV tensor).
// side_stride: bytes between K and V of a layer (tensor.stride(0) * element size).
// block_ids:   host array of tokens / block_size physical block ids.
// out[0..3]:   total_s, read_s, layer0_ms, layer_last_ms.
// Returns 0 on success, negative on error.
int kvr_load(void* hp, const char* path, int layers, int tokens, int seg_tokens, int block_size,
             unsigned long long tok_bytes, const unsigned long long* layer_ptrs,
             long long side_stride, const int* block_ids, double* out) {
  Handle* h = (Handle*)hp;
  CK(cudaSetDevice(h->device));
  Geo g = make_geo(layers, tokens, seg_tokens, block_size, tok_bytes);
  if (!geo_ok(g, tokens)) return -2;
  int fd = open(path, O_RDONLY | O_DIRECT);
  if (fd < 0) return -1;
  off_t fsz = lseek(fd, 0, SEEK_END);
  if (fsz < (off_t)g.file_bytes) { close(fd); return -3; }

  int nb = tokens / block_size;
  if (layers > h->lp_cap) {
    if (h->lp_dev) CK(cudaFree(h->lp_dev));
    CK(cudaMalloc(&h->lp_dev, layers * sizeof(char*)));
    h->lp_cap = layers;
  }
  if (nb > h->bt_cap) {
    if (h->bt_dev) CK(cudaFree(h->bt_dev));
    CK(cudaMalloc(&h->bt_dev, nb * sizeof(int)));
    h->bt_cap = nb;
  }
  CK(cudaMemcpyAsync(h->lp_dev, layer_ptrs, layers * sizeof(char*), cudaMemcpyHostToDevice, h->P.cs));
  CK(cudaMemcpyAsync(h->bt_dev, block_ids, nb * sizeof(int), cudaMemcpyHostToDevice, h->P.cs));
  CK(cudaStreamSynchronize(h->P.cs));

  RunResult r = pipeline_restore(h->P, g, fd, h->lp_dev, (long)side_stride, h->bt_dev);
  close(fd);
  if (r.total_s < 0) return -4;
  if (out) {
    out[0] = r.total_s;
    out[1] = r.read_s;
    out[2] = r.layer0_ms;
    out[3] = r.layer_last_ms;
  }
  return 0;
}

void kvr_close(void* hp) {
  Handle* h = (Handle*)hp;
  if (h->lp_dev) cudaFree(h->lp_dev);
  if (h->bt_dev) cudaFree(h->bt_dev);
  pipeline_free(h->P);
  delete h;
}

}  // extern "C"
