#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from .base import Database
import pandas as pd
from numpy import polyfit
import os
import math
from typing import TYPE_CHECKING
from collections import defaultdict

class TRACEDatabase(Database):
    def __init__(self, path: str, config: "AutoConfig"):
        self.path = path
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        #self.key_map = _KEY_MAPS[self.model]
        super().__init__(path, config)
        self.fused_kernel = False
        self.gemm = {}
        self.load()

    def load(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(self.path, "gemm_summary.csv"), sep=None, engine="python", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df = df[["batch_size", "N", "K", "top_k", "num_experts", "latency"]].apply(pd.to_numeric, errors="coerce").dropna()
        self.gemm = {
            (int(b), int(n), int(k), int(t), int(e)): float(l)
            for b, n, k, t, e, l in df.itertuples(index=False, name=None)
        }

    def lookup(self, batch_size, N, K, top_k=0, num_experts=0):
        key = (batch_size, N, K, top_k, num_experts)
        if key in self.gemm:
            return self.gemm[key]
        print("INTERPOLATING TADS")
        points = sorted(
            (k[0], v) for k, v in self.gemm.items()
            if k[1:] == (N, K, top_k, num_experts)
        )
        if not points:
            raise KeyError(key)
        if len(points) == 1:
            b0, l0 = points[0]
            return l0 * batch_size / b0

        lower = [p for p in points if p[0] < batch_size]
        upper = [p for p in points if p[0] > batch_size]

        if lower and upper:
            (b0, l0), (b1, l1) = lower[-1], upper[0]
        elif upper:
            (b0, l0), (b1, l1) = upper[0], upper[1]
        else:
            (b0, l0), (b1, l1) = lower[-2], lower[-1]

        slope = (l1 - l0) / (b1 - b0)
        return max(0.0, l0 + slope * (batch_size - b0))

    def latency(self, stage: "OpStage", batch_size: int, seq_len: int, is_prefill:bool=True) -> float:
        M = batch_size * seq_len if is_prefill else batch_size
        top_k,total_experts=0,0
        if "expert" in stage.role:
            total_experts = self.config.num_local_experts
            top_k = self.config.num_experts_per_tok
            lat = self.lookup(batch_size=M,
                              N= self.hidden_size,
                              K= self.intermediate_size,
                              top_k=top_k,
                              num_experts=total_experts)/1e3
            return lat
        _, N, K = stage.spec
        lat = self.lookup(M,N,K,top_k,total_experts)/1e3
        return lat
