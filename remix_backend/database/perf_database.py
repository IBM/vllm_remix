#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from .base import Database
from pathlib import Path
import csv
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import AutoConfig
    from src.stage import OpStage

class PerfDatabase(Database):
    def __init__(self, path:str, config:"AutoConfig", kv_cache_bytes:int=2):
        super().__init__(path, config)
        self.gemm= {}
        self.pref_attn = {}
        self.kv_cache_dtype = "bfloat16" if kv_cache_bytes ==2 else "fp8"
        self.dec_attn = {}
        self.is_moe = hasattr(config, "num_local_experts")
        self.moe = {}
        self.head_dim = int(self.config.hidden_size/ getattr(self.config, "num_attention_heads", 2))
        self.model_type = config.model_type
        self.load()


    def load(self):
        with open(os.path.join(self.path, "gemm_perf.txt"), newline="") as f:
            for row in csv.DictReader(f):
                if row["gemm_dtype"] != "bfloat16":
                    continue
                self.gemm[(int(row["m"]), int(row["n"]), int(row["k"]))] = float(row["latency"])

        with open(os.path.join(self.path, "context_attention_perf.txt"), newline="") as f:
            for row in csv.DictReader(f):
                if row["attn_dtype"] == "bfloat16" and row["kv_cache_dtype"] == self.kv_cache_dtype:
                    self.pref_attn[(int(row["batch_size"]), int(row["isl"]),
                                    int(row["num_heads"]), int(row["num_key_value_heads"]),
                                    int(row["head_dim"]))] = float(row["latency"])

        with open(os.path.join(self.path,"generation_attention_perf.txt"), newline="") as f:
            for row in csv.DictReader(f):
                if row["attn_dtype"] == "bfloat16" and row["kv_cache_dtype"] == self.kv_cache_dtype:
                    self.dec_attn[(int(row["batch_size"]), int(row["step"]),
                                int(row["num_heads"]), int(row["num_key_value_heads"]),
                                int(row["head_dim"]))] = float(row["latency"])

        if self.is_moe:
            with open(os.path.join(self.path , "moe_perf.txt"), newline="") as f:
                for row in csv.DictReader(f):
                    if row["moe_dtype"] != "bfloat16":
                        continue
                    key = (
                        int(row["num_tokens"]),
                        int(row["hidden_size"]),
                        int(row["inter_size"]),
                        int(row["topk"]),
                        int(row["num_experts"]),
                        int(row["moe_tp_size"]),
                        int(row["moe_ep_size"]),
                        row["distribution"],
                    )
                    self.moe[key] = float(row["latency"]) * 1e3
        if self.model_type in ("mamba2", "mamba", "bamba"):
            self.mixer = {}
            self.context_kernels = ["causal_conv1d_fn", "mamba_chunk_scan_combined"]
            self.generation_kernels = ["causal_conv1d_update", "selective_state_update"]
            path = Path(os.path.join(self.path , "mamba_perf.txt"))
            if path.exists():
                with open(path, newline="") as f:
                    for row in csv.DictReader(f):
                        key = (row["phase"], row["kernel_source"], int(row["batch_size"]), int(row["seq_len"]), int(row["d_state"]), int(row["d_conv"]))
                        self.mixer[key] = float(row["latency"]) * 1e3

    def moe_latency(self, num_tokens, hidden_size, inter_size, topk, num_experts,
                    moe_tp_size=1, moe_ep_size=1, distribution="power_law_1.01"):
        key = (num_tokens, hidden_size, inter_size, topk, num_experts,
               moe_tp_size, moe_ep_size, distribution)
        exact = self.moe.get(key)
        if exact is not None:
            return exact

        candidates = [
            (k, v) for k, v in self.moe.items()
            if k[1] == hidden_size
            and k[2] == inter_size
            and k[3] == topk
            and k[4] == num_experts
            and k[5] == moe_tp_size
            and k[6] == moe_ep_size
            and k[7] == distribution
        ]
        if not candidates:
            raise KeyError(f"No MoE entries for hidden={hidden_size} inter={inter_size} "
                           f"topk={topk} experts={num_experts} tp={moe_tp_size} ep={moe_ep_size} "
                           f"dist={distribution}")

        candidates.sort(key=lambda x: x[0][0])
        xs = [k[0] for k, _ in candidates]
        ys = [v for _, v in candidates]
        t = num_tokens
        if t <= xs[0]:
            return ys[0]
        if t >= xs[-1]:
            return ys[-1]
        i = next(i for i in range(len(xs) - 1) if xs[i] <= t <= xs[i + 1])
        alpha = (t - xs[i]) / (xs[i + 1] - xs[i])
        return ys[i] + alpha * (ys[i + 1] - ys[i])

    def gemm_interpolate(self, m, n, k):
        exact = self.gemm.get((m, n, k))
        if exact is not None:
            return exact

        query = (m, n, k)
        missing = []
        for i, dim in enumerate(query):
            present = set(key[i] for key in self.gemm)
            if dim not in present:
                missing.append((i, sorted(present)))

        if len(missing) > 1:
            raise ValueError(f"Cannot interpolate >1 missing dim for (m={m}, n={n}, k={k})")

        def interp(fixed, idx, v):
            fixed.sort(key=lambda x: x[0][idx])
            xs = [key[idx] for key, _ in fixed]
            lats = [lat for _, lat in fixed]
            if v <= xs[0]:
                return lats[0]
            if v >= xs[-1]:
                return lats[-1]
            i = next(i for i in range(len(xs) - 1) if xs[i] <= v <= xs[i + 1])
            alpha = (v - xs[i]) / (xs[i + 1] - xs[i])
            return lats[i] + alpha * (lats[i + 1] - lats[i])

        if len(missing) == 1:
            idx, _ = missing[0]
            fixed = [(key, lat) for key, lat in self.gemm.items()
                     if all(key[j] == query[j] for j in range(3) if j != idx)]
            if not fixed:
                raise ValueError(f"No entries to interpolate (m={m}, n={n}, k={k})")
            print("interpolating", m,n,k)
            return interp(fixed, idx, query[idx])

        fixed = [(key, lat) for key, lat in self.gemm.items() if key[0] == m and key[1] == n]
        if not fixed:
            raise ValueError(f"No entries for (m={m}, n={n}, *)")
        return interp(fixed, 2, k)

    def gemm_latency(self, m, n, k):
        return self.gemm_interpolate(m, n, k) * 1e3

    def mixer_latency(self, batch_size, seq_len, is_prefill):
        dim_key = 1
        phase = "generation"
        if is_prefill:
            dim_key = seq_len
            phase = "context"
        d_state = getattr(self.config, "mamba_d_state", getattr(self.config, "state_size", None))
        d_conv = getattr(self.config, "mamba_d_conv", getattr(self.config, "conv_kernel", None))
        if phase=="generation":
            keys = [(phase, kernel, batch_size, dim_key, d_state, d_conv) for kernel in self.generation_kernels]
        else:
            keys = [(phase, kernel, batch_size, dim_key, d_state, d_conv) for kernel in self.context_kernels]
        latency = 0.0
        for key in keys:
            #key = (row["phase"], int(row["batch_size"]), int(row["seq_len"]), int(row["d_state"]), int(row["d_conv"]))
            exact = self.mixer.get(key)
            if exact is not None:
                latency += exact
            else:
                raise ValueError(f"No Mixer value for {key}")
        return latency


    def attn_latency(self, batch_size, num_heads, head_dim, num_kv_heads, seq_len, is_prefill):
        if is_prefill:
            table = self.pref_attn
            dim_key = seq_len
        else:
            table = self.dec_attn
            dim_key = seq_len - 1 if seq_len != 1 else 1

        key = (batch_size, dim_key, num_heads, num_kv_heads, head_dim)
        exact = table.get(key)
        if exact is not None:
            if exact == 0.0:
                print("Zero attention")
                print(key)
            return exact * 1e3
        candidates = [
            (k, v) for k, v in table.items()
            if k[1] == dim_key and k[2] == num_heads and k[3] == num_kv_heads and k[4] == head_dim
        ]
        if not candidates:
            return float("inf")

        candidates.sort(key=lambda x: x[0][0])
        xs = [k[0] for k, _ in candidates]
        ys = [v for _, v in candidates]
        if batch_size <= xs[0]:
            # below range: linear extrapolation from the two smallest points
            if len(xs) >= 2:
                alpha = (batch_size - xs[0]) / (xs[1] - xs[0])
                return (ys[0] + alpha * (ys[1] - ys[0])) * 1e3
            return ys[0] * 1e3
        if batch_size >= xs[-1]:
            # above range: linear extrapolation from the two largest points
            if len(xs) >= 2:
                alpha = (batch_size - xs[-2]) / (xs[-1] - xs[-2])
                return (ys[-2] + alpha * (ys[-1] - ys[-2])) * 1e3
            return ys[-1] * 1e3
        i = next(i for i in range(len(xs) - 1) if xs[i] <= batch_size <= xs[i + 1])
        alpha = (batch_size - xs[i]) / (xs[i + 1] - xs[i])
        return (ys[i] + alpha * (ys[i + 1] - ys[i])) * 1e3

    def latency(self, stage: "OpStage", batch_size: int, seq_len: int, is_prefill:bool=True) -> float:
        if stage.role == "attn":
            return self.attn_latency(batch_size, self.config.num_attention_heads, self.head_dim, self.config.num_key_value_heads, seq_len, is_prefill)
        if stage.role == "mixer":
            return self.mixer_latency(batch_size, seq_len, is_prefill)

        M = batch_size * seq_len if is_prefill else batch_size
        if "expert" in stage.role:
            total_experts = self.config.num_local_experts
            top_k = self.config.num_experts_per_tok
            return self.moe_latency(M, self.config.hidden_size, self.config.intermediate_size,
                                    top_k, total_experts, moe_tp_size=1, moe_ep_size=1)
        _, N, K = stage.spec
        if N == 0 and K == 0:
            return 0.0
        return self.gemm_latency(M, N, K)
