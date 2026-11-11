#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from .base import Database
import os
import csv
import math

ATTACC_CONTEXT_STEP = 16


def round_context_to_attacc_step(seq_len, step=ATTACC_CONTEXT_STEP):
    anchor = step
    if seq_len <= anchor:
        return anchor
    k = round((seq_len - anchor) / step)
    return anchor + k * step

class AttaccDatabase(Database):
    def __init__(self, path: str, config: "AutoConfig", kv_cache_bytes:int = 2):
        super().__init__(path, config)
        self.attn_data = {}
        self.num_heads =  getattr(config, "num_attention_heads", 0) # no GQA for ATTACC
        self.head_dim = getattr(config, "head_dim", None)
        if self.head_dim is None and self.num_heads:
            self.head_dim = getattr(config, "hidden_size")//self.num_heads
        self.kv_cache_bytes = kv_cache_bytes
        self.load()

    def load(self):
        attn_path = os.path.join(self.path, "attn_profile.csv")
        mvm_path = os.path.join(self.path, "mvm_profile.csv")
        if attn_path is not None:
            with open(attn_path, newline="") as f:
                for row in csv.DictReader(f):
                    is_prefill = "prefill" in row["sweep"].lower()
                    if row["dbyte"] == str(self.kv_cache_bytes):
                        key = (is_prefill,
                            int(row["batch_size"]),
                            int(row["seq_len"]),
                            int(row["num_heads"]),
                            int(row["head_dim"]))
                        self.attn_data[key] = float(row["latency_us"])
        with open(mvm_path, newline="") as f:
            for row in csv.DictReader(f):
                key = (int(row["batch_size"]), int(row["input_dim"]), int(row["output_dim"]))
                self.data[key] = float(row["latency_us"])

    def attn_latency(self, batch_size, seq_len, is_prefill):
        if is_prefill:
            res = self.attn_data[(is_prefill, batch_size, seq_len, self.num_heads, self.head_dim)]
        else:
            res = self.attn_data[(is_prefill, batch_size, seq_len-1, self.num_heads, self.head_dim)]
        if not bool(res):
            return math.inf
        #print(batch_size, seq_len, is_prefill, res)
        return res

    def gemm_latency(self, batch_size, output_dim, input_dim):
        return self.data[(batch_size, input_dim, output_dim)]

    def latency(self, stage: "Stage", batch_size: int, seq_len: int, is_prefill:bool=True) -> float:
        if stage.role == "attn":
            seq_len = round_context_to_attacc_step(seq_len)
            return self.attn_latency(batch_size = batch_size, 
                              seq_len = seq_len,
                              is_prefill = is_prefill)
        _, N, K = stage.spec
        M = batch_size * seq_len if is_prefill else batch_size
        res = self.gemm_latency(M, N, K)
        if not bool(res):
            return math.inf
        return res

