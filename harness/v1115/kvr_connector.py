"""KvrSSDConnector — vLLM 0.10.1.1 v1 KV connector backed by libkvr.so (P2).

Store: when a new request's whole prompt is prefilled in one step, save_kv_layer writes each
layer of the first L = floor(len / seg_tokens) * seg_tokens tokens into one file per prefix,
in the layout kvr_core.cuh reads (layer-major; per 128-token item: K tokens, then V tokens).
Load: get_num_new_matched_tokens finds the longest stored prefix L <= len(prompt) - 1 by a
chained hash over 128-token chunks; start_load_kv restores it synchronously through
libkvr.so (parallel O_DIRECT reads + pinned ring + H2D + scatter into paged KV).

Load with:
  KVTransferConfig(kv_connector="KvrSSDConnector", kv_role="kv_both",
                   kv_connector_module_path="kvr_connector",
                   kv_connector_extra_config={"kvr_dir": ..., "kvr_lib": ".../libkvr.so"})
Per-request options: SamplingParams(extra_args={"kv_transfer_params": {...}}) with
  "kvr_skip": True   neither load nor store (recompute reference)
  "kvr_verify": True on store, record per-layer checksums; on load, check them (slow).
"""
import ctypes
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class _Op:
    req_id: str
    path: str
    tokens: int
    block_ids: list[int]
    verify: bool = False


@dataclass
class KvrMetadata(KVConnectorMetadata):
    loads: list[_Op] = field(default_factory=list)
    stores: list[_Op] = field(default_factory=list)


def _params(sampling_params) -> dict:
    ea = getattr(sampling_params, "extra_args", None) or {}
    return ea.get("kv_transfer_params") or {}


def _skip(request) -> bool:
    p = getattr(request, "kv_transfer_params", None)
    return bool(p and p.get("kvr_skip"))


class KvrSSDConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        cfg = vllm_config.kv_transfer_config
        get = cfg.get_from_extra_config
        self._dir = get("kvr_dir", "/tmp/kvr")
        self._lib_path = get("kvr_lib", "libkvr.so")
        self._seg = int(get("kvr_seg_tokens", 128))
        self._threads = int(get("kvr_threads", 8))
        self._qd = int(get("kvr_qd", 16))
        self._batch = int(get("kvr_batch", 32))
        self._ring_mb = int(get("kvr_ring_mb", 128))
        self._copy = int(get("kvr_copy_mode", 1))
        self._log = get("kvr_log", None)
        self._block = vllm_config.cache_config.block_size
        assert self._seg % self._block == 0
        os.makedirs(self._dir, exist_ok=True)
        # scheduler side
        self._match: dict[str, tuple[int, str]] = {}
        self._need_load: dict[str, tuple[int, str]] = {}
        # worker side
        self._lib = None
        self._h = None
        self._layers: list[torch.Tensor] = []
        self._layer_idx: dict[str, int] = {}
        self._store_fds: dict[str, int] = {}
        self._sums: dict[str, list[int]] = {}

    # ------------------------------------------------------------ keys
    def _chain(self, tokens: list[int], max_tokens: int) -> list[tuple[int, str]]:
        """(L, path) for every L = k * seg <= max_tokens, shortest first."""
        arr = np.asarray(tokens[:max_tokens], dtype=np.int64)
        out, h = [], b""
        for k in range(len(arr) // self._seg):
            h = hashlib.md5(h + arr[k * self._seg:(k + 1) * self._seg].tobytes(),
                            usedforsecurity=False).digest()
            L = (k + 1) * self._seg
            out.append((L, os.path.join(self._dir, f"{h.hex()}_{L}.kv")))
        return out

    # ------------------------------------------------------------ scheduler side
    def get_num_new_matched_tokens(self, request: "Request",
                                   num_computed_tokens: int) -> tuple[int, bool]:
        if _skip(request):
            return 0, False
        for L, path in reversed(self._chain(request.prompt_token_ids,
                                            len(request.prompt_token_ids) - 1)):
            if os.path.exists(path):
                if L <= num_computed_tokens:
                    return 0, False
                self._match[request.request_id] = (L, path)
                return L - num_computed_tokens, False
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        m = self._match.pop(request.request_id, None)
        if num_external_tokens > 0 and m is not None:
            self._need_load[request.request_id] = m

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = KvrMetadata()
        for r in scheduler_output.scheduled_new_reqs:
            bids = list(r.block_ids[0])
            prm = _params(r.sampling_params)
            verify = bool(prm.get("kvr_verify"))
            if r.req_id in self._need_load:
                L, path = self._need_load.pop(r.req_id)
                meta.loads.append(_Op(r.req_id, path, L, bids[:L // self._block], verify))
                continue
            n = len(r.prompt_token_ids)
            L = n // self._seg * self._seg
            if L == 0 or prm.get("kvr_skip") or r.num_computed_tokens != 0:
                continue
            if scheduler_output.num_scheduled_tokens.get(r.req_id, 0) < n:
                logger.warning("kvr: prompt of %s is chunked; not storing "
                               "(raise max_num_batched_tokens)", r.req_id)
                continue
            path = self._chain(r.prompt_token_ids, L)[-1][1]
            if not os.path.exists(path):
                meta.stores.append(_Op(r.req_id, path, L, bids[:L // self._block], verify))
        # loads for preempted-and-resumed requests are not supported (not needed here)
        assert not self._need_load, f"kvr: loads not scheduled: {list(self._need_load)}"
        return meta

    # ------------------------------------------------------------ worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        def idx(name: str) -> int:
            m = re.search(r"layers\.(\d+)\.", name)
            assert m, name
            return int(m.group(1))

        names = sorted(kv_caches, key=idx)
        self._layer_idx = {n: i for i, n in enumerate(names)}
        self._layers = [kv_caches[n] for n in names]
        t = self._layers[0]
        assert t.dim() == 5 and t.shape[0] == 2 and t.shape[2] == self._block, (
            f"kvr expects FlashAttention layout (2, blocks, block_size, heads, dim), got {t.shape}")
        for x in self._layers:
            assert x.shape == t.shape and x.dtype == t.dtype
            assert x[0].is_contiguous() and x[1].is_contiguous()
            assert x.stride(0) == t.stride(0)
        self._tok_bytes = t.shape[3] * t.shape[4] * t.element_size()
        self._side_stride = t.stride(0) * t.element_size()
        self._ptrs = (ctypes.c_ulonglong * len(self._layers))(*[x.data_ptr() for x in self._layers])
        lib = ctypes.CDLL(self._lib_path)
        lib.kvr_open.restype = ctypes.c_void_p
        lib.kvr_open.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_ulonglong, ctypes.c_int]
        lib.kvr_load.restype = ctypes.c_int
        lib.kvr_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong,
                                 ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_longlong,
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
        self._lib = lib
        self._h = lib.kvr_open(t.device.index, self._threads, self._qd, self._batch,
                               self._ring_mb << 20, self._copy)
        logger.info("kvr: %d layers, kv shape %s %s, tok_bytes %d, side_stride %d, lib %s",
                    len(self._layers), tuple(t.shape), t.dtype, self._tok_bytes,
                    self._side_stride, self._lib_path)

    def _meta(self) -> Optional[KvrMetadata]:
        m = self._connector_metadata
        return m if isinstance(m, KvrMetadata) else None

    def _gather(self, layer: torch.Tensor, block_ids: list[int], tokens: int) -> torch.Tensor:
        """[S, 2, seg, heads*dim] in file order for one layer."""
        b = torch.as_tensor(block_ids, device=layer.device)
        S = tokens // self._seg
        k = layer[0, b].reshape(S, self._seg, -1)
        v = layer[1, b].reshape(S, self._seg, -1)
        return torch.stack([k, v], dim=1)

    @staticmethod
    def _checksum(x: torch.Tensor) -> int:
        w = x.contiguous().view(torch.int32).reshape(-1).to(torch.int64)
        pos = torch.arange(w.numel(), device=w.device, dtype=torch.int64) % 65521 + 1
        return int((w * pos).sum().item())

    def _append_log(self, rec: dict[str, Any]):
        if self._log:
            with open(self._log, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        meta = self._meta()
        if meta is None or not meta.loads:
            return
        torch.cuda.current_stream().synchronize()
        for op in meta.loads:
            bids = (ctypes.c_int * len(op.block_ids))(*op.block_ids)
            out = (ctypes.c_double * 4)()
            t0 = time.perf_counter()
            rc = self._lib.kvr_load(self._h, op.path.encode(), len(self._layers), op.tokens,
                                    self._seg, self._block, self._tok_bytes, self._ptrs,
                                    self._side_stride, bids, out)
            wall = time.perf_counter() - t0
            if rc != 0:
                raise RuntimeError(f"kvr_load({op.path}) failed: {rc}")
            rec = dict(event="load", req=op.req_id, tokens=op.tokens, wall_s=wall,
                       pipe_s=out[0], read_s=out[1], layer0_ms=out[2], last_ms=out[3])
            if op.verify:
                ref = json.load(open(op.path + ".sum"))
                got = [self._checksum(self._gather(x, op.block_ids, op.tokens))
                       for x in self._layers]
                rec["verify_ok"] = got == ref
                if got != ref:
                    bad = [i for i, (a, b) in enumerate(zip(got, ref)) if a != b]
                    raise RuntimeError(f"kvr verify failed on layers {bad[:8]}")
            self._append_log(rec)
            logger.info("kvr: loaded %d tokens in %.4f s (pipeline %.4f s)", op.tokens,
                        wall, out[0])

    def wait_for_layer_load(self, layer_name: str) -> None:
        return  # load is synchronous in start_load_kv

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        meta = self._meta()
        if meta is None or not meta.stores or torch.cuda.is_current_stream_capturing():
            return
        li = self._layer_idx[layer_name]
        for op in meta.stores:
            x = self._gather(kv_layer, op.block_ids, op.tokens)
            if op.req_id not in self._store_fds:
                self._store_fds[op.req_id] = os.open(op.path + ".tmp",
                                                     os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                self._sums[op.req_id] = [0] * len(self._layers)
            if op.verify:
                self._sums[op.req_id][li] = self._checksum(x)
            buf = x.contiguous().view(torch.int16).cpu().numpy()
            os.pwrite(self._store_fds[op.req_id], buf.tobytes(), li * buf.nbytes)

    def wait_for_save(self):
        meta = self._meta()
        if meta is None:
            return
        for op in meta.stores:
            fd = self._store_fds.pop(op.req_id, None)
            if fd is None:
                continue
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            if op.verify:
                with open(op.path + ".sum", "w") as f:
                    json.dump(self._sums.pop(op.req_id), f)
            os.rename(op.path + ".tmp", op.path)
            self._append_log(dict(event="store", req=op.req_id, tokens=op.tokens,
                                  path=op.path))
