#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import csv
import math
import os

from .base_backend import BaseBackend

ATTACC_CONTEXT_STEP = 16


def round_context_to_attacc_step(seq_len, step=ATTACC_CONTEXT_STEP):
    anchor = step
    if seq_len <= anchor:
        return anchor
    k = round((seq_len - anchor) / step)
    return anchor + k * step


class AttAccBackend(BaseBackend):
    def __init__(self, path='./remix_backend/data/attacc_database/', num_kv_heads=None, head_dim=None):
        self.path = path
        self.attn_data = {}
        self.attn_submods = {}
        self.default_num_kv_heads = num_kv_heads
        self.default_head_dim = head_dim
        self.req_seq_lens = {}
        self._last_scheduler_output = None
        self._step_decode_lens = []
        self.load()

    def load(self):
        attn_path = os.path.join(self.path, "attn_profile.csv")
        with open(attn_path, newline="") as f:
            for row in csv.DictReader(f):
                is_prefill = "prefill" in row["sweep"].lower()
                key = (
                    is_prefill,
                    int(row["batch_size"]),
                    int(row["seq_len"]),
                    int(row["num_heads"]),
                    int(row["head_dim"]),
                )
                self.attn_data[key] = float(row["latency_us"])

    def register_submod(self, name, gm):
        if not hasattr(gm, "graph"):
            return False
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if "unified_attention" not in str(node.target):
                continue
            self.attn_submods[name] = self._extract_geometry(node)
            return True
        return False

    def _extract_geometry(self, node):
        num_kv_heads = self.default_num_kv_heads
        head_dim = self.default_head_dim
        try:
            k_shape = node.args[1].meta["example_value"].shape
            if len(k_shape) >= 3:
                num_kv_heads = int(k_shape[-2]) * 4
                head_dim = int(k_shape[-1])
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        return num_kv_heads, head_dim

    def backend(self, graph_module, example_inputs):
        return graph_module

    def _register_new_requests(self, scheduler_output):
        for req in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            req_id = getattr(req, "req_id", None)
            if req_id is None:
                continue
            base = getattr(req, "num_computed_tokens", 0) or 0
            self.req_seq_lens[req_id] = int(base)

    def _retire_requests(self, scheduler_output):
        finished = set(getattr(scheduler_output, "finished_req_ids", None) or ())
        preempted = getattr(scheduler_output, "preempted_req_ids", None)
        if preempted:
            finished |= set(preempted)
        for req_id in finished:
            self.req_seq_lens.pop(req_id, None)

    def _advance_step(self, scheduler_output):
        self._register_new_requests(scheduler_output)

        decode_lens = []
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        for req_id, n_tokens in num_scheduled.items():
            n_tokens = int(n_tokens)
            if n_tokens <= 0:
                continue
            prev = self.req_seq_lens.get(req_id, 0)
            new_len = prev + n_tokens
            self.req_seq_lens[req_id] = new_len
            if n_tokens == 1:
                decode_lens.append(new_len)

        self._retire_requests(scheduler_output)
        return decode_lens

    def attn_latency(self, batch_size, num_heads, head_dim, seq_len, is_prefill):
        lookup = round_context_to_attacc_step(seq_len)
        res = self.attn_data.get((is_prefill, batch_size, lookup, num_heads, head_dim))
        if res is not None:
            return math.inf if not res else res

        candidates = sorted(
            (b, s)
            for (p, b, s, h, d) in self.attn_data
            if p == is_prefill and h == num_heads and d == head_dim
            and b >= batch_size and s >= lookup
        )
        if not candidates:
            raise KeyError(
                f"no attacc entry for prefill={is_prefill} bs>={batch_size} "
                f"seq_len>={lookup} heads={num_heads} head_dim={head_dim}"
            )
        res = self.attn_data[(is_prefill, candidates[0][0], candidates[0][1], num_heads, head_dim)]
        return math.inf if not res else res

    def return_latency(self, profiler, meta_ctx) -> float:
        geom = self.attn_submods.get(
            (name[name.find("submod"):] if (name := getattr(meta_ctx, "name", None)) else None)
        )
        if geom is None:
            return 0.0
        num_kv_heads, head_dim = geom
        if num_kv_heads is None or head_dim is None:
            return 0.0

        scheduler_output = getattr(meta_ctx, "scheduler_output", None)
        if scheduler_output is None:
            return 0.0

        if scheduler_output is not self._last_scheduler_output:
            self._last_scheduler_output = scheduler_output
            self._step_decode_lens = self._advance_step(scheduler_output)

        if not self._step_decode_lens:
            return 0.0

        groups = {}
        for seq_len in self._step_decode_lens:
            key = round_context_to_attacc_step(seq_len)
            groups[key] = groups.get(key, 0) + 1

        latency = max(
            self.attn_latency(count, num_kv_heads, head_dim, rounded_len, False)
            for rounded_len, count in groups.items()
        )
        return latency * 1e-6

    def move_to_device(self, x):
        return x.cuda()