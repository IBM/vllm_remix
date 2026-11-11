#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import json
import math
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from math import gcd
import heapq
from bisect import bisect_right
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.fx as fx

BYTES_PER_ELEMENT = 2

def _unwrap_dynamo_path(path: str) -> str:
    if not path:
        return ""
    path = re.sub(r"_modules\['([^']*)'\]", r"\1", path)
    path = re.sub(r"^L\[[^\]]*\]\.?", "", path)
    path = re.sub(r"^self\.?", "", path)
    return path.strip(".")


def module_key(node) -> str:
    stack = node.meta.get("nn_module_stack") or {}
    vals = list(stack.values())
    if not vals:
        return ""
    entry = vals[-1]
    raw = entry[0] if isinstance(entry, (tuple, list)) else entry
    return _unwrap_dynamo_path(str(raw))


def module_cls(node) -> str:
    stack = node.meta.get("nn_module_stack") or {}
    vals = list(stack.values())
    if not vals:
        return ""
    entry = vals[-1]
    if isinstance(entry, (tuple, list)) and len(entry) > 1:
        t = entry[1]
        return getattr(t, "__name__", str(t)).split(".")[-1]
    return ""


def access_config(config, *names, default=None):
    for c in (config, getattr(config, "text_config", None),
              getattr(config, "hf_text_config", None),
              getattr(config, "hf_config", None)):
        if c is None:
            continue
        for n in names:
            v = getattr(c, n, None)
            if v is not None:
                return v
    return default


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def kv_bytes_per_layer(config, batch_size, seq_len, dtype_bytes=BYTES_PER_ELEMENT):
    nh = access_config(config, "num_attention_heads", "n_head")
    nkv = access_config(config, "num_key_value_heads", "num_kv_heads", "n_head_kv", default=nh)
    hidden = access_config(config, "hidden_size", "d_model")
    hd = access_config(config, "head_dim")
    if hd is None and hidden and nh:
        hd = hidden // nh
    if not (nkv and hd):
        return 0
    return int(2 * nkv * hd * batch_size * seq_len * dtype_bytes)


def layer_index(name):
    m = re.search(r"layers\.(\d+)(?:\.|$)", name or "")
    return int(m.group(1)) if m else None


def attn_layer_set(config):
    if config is None:
        return None
    idx = access_config(config, "attn_layer_indices", "attention_layer_indices")
    if idx is not None:
        return {int(x) for x in idx}
    types = access_config(config, "layers_block_type", "layer_types")
    if isinstance(types, (list, tuple)):
        return {i for i, t in enumerate(types)
                if "attn" in str(t).lower() or "attention" in str(t).lower()}
    mt = str(access_config(config, "model_type", default="")).lower()
    if mt in ("mamba", "mamba2", "falcon_mamba"):
        return set()
    return None


def make_kv_filter(config):
    layers = attn_layer_set(config)
    if layers is None:
        return None

    def f(st):
        li = layer_index(st.name)
        if li is None:
            return bool(layers)
        return li in layers
    return f


def ssm_state_bytes(config, batch_size, dtype_bytes=BYTES_PER_ELEMENT):
    hidden = _int_or_none(access_config(config, "hidden_size", "d_model"))
    expand = access_config(config, "mamba_expand", "expand")
    d_state = _int_or_none(access_config(config, "mamba_d_state", "state_size"))
    d_conv = _int_or_none(access_config(config, "mamba_d_conv", "conv_kernel"))
    if not (hidden and expand and d_state and d_conv):
        return 0
    d_inner = int(float(expand) * hidden)
    n_heads = _int_or_none(access_config(config, "mamba_n_heads", "num_heads"))
    head_dim = _int_or_none(access_config(config, "mamba_d_head", "head_dim"))
    if n_heads and head_dim:
        n_groups = _int_or_none(access_config(config, "mamba_n_groups", "n_groups", default=1)) or 1
        conv_width = d_inner + 2 * n_groups * d_state
        ssm = batch_size * n_heads * head_dim * d_state
    else:
        conv_width = d_inner
        ssm = batch_size * d_inner * d_state
    conv = batch_size * conv_width * (d_conv - 1)
    return int((conv + ssm) * dtype_bytes)


def get_node_val(node):
    v = node.meta.get("val", None)
    if v is None:
        v = node.meta.get("example_value", None)
    return v


def _tensors(node):
    v = get_node_val(node)
    if isinstance(v, torch.Tensor):
        return [v]
    if isinstance(v, (list, tuple)):
        return [x for x in v if isinstance(x, torch.Tensor)]
    return []

def _concrete(d):
    return d if isinstance(d, int) and not isinstance(d, bool) else None

def _shapes(node):
    return [tuple(_concrete(d) for d in t.shape) for t in _tensors(node)]


def _elem_size(t, default=BYTES_PER_ELEMENT):
    dt = getattr(t, "dtype", None)
    n = getattr(dt, "itemsize", None)
    if isinstance(n, int) and n > 0:
        return n
    return default


def _is_param(node, static_placeholders=True):
    if not hasattr(node, "op"):
        return False
    if node.op == "get_attr":
        return True
    if node.op != "placeholder":
        return False
    v = get_node_val(node)
    if isinstance(v, torch.nn.Parameter):
        return True
    if not static_placeholders:
        return False
    ts = _tensors(node)
    if not ts:
        return False
    return all(len(t.shape) >= 1 and not any(_concrete(d) is None for d in t.shape)
               for t in ts)


def _numel(shape, toks):
    n = 1
    for d in shape:
        n *= d if d is not None else int(toks)
    return n


def _static_numel(shape):
    n = 1
    for d in shape:
        if d is None:
            return None
        n *= d
    return n


def _dyn_numel(shapes, toks):
    return sum(_numel(s, toks) for s in shapes)


def compile_device_map(device_map: Dict[str, List[str]]):
    exact, wildcard = {}, []
    for k, v in (device_map or {}).items():
        devs = v if isinstance(v, (list, tuple)) else [v]
        if any(c in k for c in "*?[]()|+^$\\"):
            pat = k.replace(".", r"\.").replace("*", ".*").replace("?", ".")
            wildcard.append((re.compile("^" + pat + "$"), list(devs)))
        else:
            exact[k] = list(devs)
    return exact, wildcard


@dataclass
class StageEdge:
    producer: Optional[int]
    shapes: List[tuple] = field(default_factory=list)

    def payload(self, toks):
        return _dyn_numel(self.shapes, toks)


PRICED_ROLES = {"attn", "mixer", "moe_expert", "moe_gate"}

HEAVY_OPS = ("mm", "bmm", "addmm", "matmul", "linear", "conv", "einsum",
             "scaled_dot_product", "attention", "attn", "scaled_mm")


def _is_heavy(node):
    t = str(getattr(node, "target", "")).lower()
    return any(h in t for h in HEAVY_OPS)


ROLE_ALIASES = {
    "self_attn": "attn",
    "attn": "attn",
    "attention": "attn",
    "mixer": "mixer",
    "mamba": "mixer",
    "mamba_mixer": "mixer",
    "experts": "moe_expert",
    "gate": "moe_gate",
}


@dataclass
class Stage:
    """
    Stage refers to nodes that carry a computational primitve
    i.e. GEMMs, Attention, SSM
    """
    name: str
    role: str
    cls: str = ""
    candidates: List[str] = field(default_factory=list)
    nodes: List[Any] = field(default_factory=list)
    inputs: List[StageEdge] = field(default_factory=list)
    out_shapes: List[tuple] = field(default_factory=list)
    param_bytes: int = 0
    weight_shapes: List[tuple] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    index: int = 0
    spec: Optional[tuple] = None
    members: List[str] = field(default_factory=list)
    out_elems: int = 0
    is_parent_module: bool = False

    def gemm_shape(self):
        if self.nodes and not any(_is_heavy(n) for n in self.nodes) \
                and any(getattr(n, "target", None) is not None for n in self.nodes):
            return None
        mats = [s for s in self.weight_shapes if len(s) == 2]
        if not mats:
            return None
        N = sum(_static_numel((s[0],)) or 0 for s in mats)
        K = _static_numel((mats[0][1],))
        if not N or not K:
            return None
        return (N, K)

    def attach_dims(self, toks):
        self.out_elems = _dyn_numel(self.out_shapes, toks)
        nk = self.gemm_shape()
        self.spec = (toks,) + nk if nk else (toks, 0, 0)


class DAG:
    """
    LLM DAG abstraction of one stage per computational primitive.
    Data flow from the original torch.fx Graph is maintained.
    """
    def __init__(self, graph: fx.GraphModule, device_map, default_device,
                 role_map=None, bytes_per_element=BYTES_PER_ELEMENT,
                 absorb=True, priced_roles=None, static_placeholders=True,
                 glue_bytes=1 << 20):
        self.gm = graph
        self.static_placeholders = static_placeholders
        self.absorb = absorb
        self.priced_roles = set(priced_roles if priced_roles is not None else PRICED_ROLES)
        self.exact, self.wildcard = compile_device_map(device_map)
        self.default_device = default_device
        self.role_map = role_map or {}
        self.bpe = bytes_per_element
        self._heavy_seen = True
        self.glue_bytes = glue_bytes
        self.stages: List[Stage] = []
        self.node_stage: Dict[Any, int] = {}
        self._build()

    def _param(self, node):
        return _is_param(node, self.static_placeholders)

    def _lookup(self, name):
        if name in self.exact:
            return self.exact[name]
        for rgx, devs in self.wildcard:
            if rgx.match(name):
                return devs
        return None

    def _anchor(self, key):
        cur = key
        while cur:
            devs = self._lookup(cur)
            if devs:
                return cur, list(devs)
            if "." not in cur:
                break
            cur = cur.rsplit(".", 1)[0]
        return key, None

    def _role(self, key, cls):
        leaf = (key.rsplit(".", 1)[-1] if key else "").lower()
        for k in (key, leaf, cls):
            if k and k in self.role_map:
                return self.role_map[k]
        if leaf in ROLE_ALIASES:
            return ROLE_ALIASES[leaf]
        return leaf or (cls or "other").lower()

    def _build(self):
        groups, order = self._group()
        groups = self._absorb(groups, order)
        self._materialize(groups, order)

    def _group(self):
        nodes = [n for n in self.gm.graph.nodes
                 if n.op in ("call_function", "call_method", "call_module")
                 and not self._param(n)]
        order = {n: i for i, n in enumerate(self.gm.graph.nodes)}

        by_key = OrderedDict()
        home = {}
        for n in nodes:
            key = module_key(n)
            if not key:
                continue
            anchor, devs = self._anchor(key)
            if anchor not in by_key:
                by_key[anchor] = {"cls": module_cls(n), "devs": devs,
                                  "nodes": [], "keys": OrderedDict()}
            by_key[anchor]["nodes"].append(n)
            by_key[anchor]["keys"][key] = None
            home[n] = anchor

        if not by_key:
            by_key["__root__"] = {"cls": "", "devs": None, "nodes": list(nodes),
                                  "keys": OrderedDict()}
            for n in nodes:
                home[n] = "__root__"

        loose = [n for n in nodes if n not in home]
        for _ in range(3):
            if not loose:
                break
            rest = []
            for n in loose:
                h = None
                for a in n.all_input_nodes:
                    if a in home:
                        h = home[a]
                for u in n.users:
                    if h is None and u in home:
                        h = home[u]
                if h is None:
                    rest.append(n)
                else:
                    home[n] = h
                    by_key[h]["nodes"].append(n)
            if len(rest) == len(loose):
                break
            loose = rest
        for n in loose:
            by_key.setdefault("__root__", {"cls": "", "devs": None, "nodes": [],
                                           "keys": OrderedDict()})
            by_key["__root__"]["nodes"].append(n)
            home[n] = "__root__"

        return self._split_containers(by_key, order), order

    @staticmethod
    def _base(key):
        return key.split("#", 1)[0]

    def _split_containers(self, by_key, order):
        keys = list(by_key)
        out = OrderedDict()
        for k, info in by_key.items():
            desc = sorted(order[n] for o in keys
                          if o != k and o.startswith(k + ".")
                          for n in by_key[o]["nodes"])
            if not desc:
                out[k] = info
                continue
            segs = [[]]
            for n in sorted(info["nodes"], key=lambda x: order[x]):
                if segs[-1]:
                    lo, hi = order[segs[-1][-1]], order[n]
                    j = bisect_right(desc, lo)
                    if j < len(desc) and desc[j] < hi:
                        segs.append([])
                segs[-1].append(n)
            for s_i, seg in enumerate(segs):
                name = k if s_i == 0 else f"{k}#{s_i}"
                out[name] = {"cls": info["cls"], "devs": info["devs"], "nodes": seg,
                             "keys": OrderedDict(info["keys"])}
        return out

    def _transparent(self, key, info, order, containers=()):
        if info.get("devs"):
            return False
        role = self._role(self._base(key), info["cls"])
        if role in self.priced_roles and key not in containers:
            return False
        if key not in containers and self._group_param_bytes(info) > self.glue_bytes:
            return False
        if self._heavy_seen:
            return not any(_is_heavy(n) for n in info["nodes"])
        for n in info["nodes"]:
            for a in n.all_input_nodes:
                if self._param(a) and any(len(sh) >= 2 for sh in _shapes(a)):
                    return False
        return True

    def _group_param_bytes(self, info):
        own = set(info["nodes"])
        seen, total = set(), 0
        for n in info["nodes"]:
            for a in n.all_input_nodes:
                if a in own or id(a) in seen or not self._param(a):
                    continue
                seen.add(id(a))
                for t in _tensors(a):
                    sn = _static_numel(tuple(_concrete(d) for d in t.shape))
                    if sn:
                        total += sn * _elem_size(t, self.bpe)
        return total

    @classmethod
    def _containers(cls, groups):
        bases = {k: cls._base(k) for k in groups}
        out = set()
        for k, b in bases.items():
            for o, ob in bases.items():
                if ob != b and ob.startswith(b + "."):
                    out.add(k)
                    break
        return out

    def _group_graph(self, groups, home):
        succ = {k: set() for k in groups}
        for k, info in groups.items():
            for n in info["nodes"]:
                for u in n.users:
                    h = home.get(u)
                    if h is not None and h != k:
                        succ[k].add(h)
        return succ

    @staticmethod
    def _indirect_path(succ, src, dst):
        stack = [x for x in succ[src] if x != dst]
        seen = set(stack)
        while stack:
            x = stack.pop()
            for y in succ[x]:
                if y == dst:
                    return True
                if y != src and y not in seen:
                    seen.add(y)
                    stack.append(y)
        return False

    def _creates_cycle(self, succ, a, b):
        return self._indirect_path(succ, a, b) or self._indirect_path(succ, b, a)

    @staticmethod
    def _merge_succ(succ, key, tgt):
        out = succ.pop(key)
        succ[tgt] |= out
        succ[tgt].discard(tgt)
        succ[tgt].discard(key)
        for k, s in succ.items():
            if key in s:
                s.discard(key)
                if k != tgt:
                    s.add(tgt)

    def _targets(self, info, own, prod, cons, groups, order):
        first = {k: min(order[n] for n in groups[k]["nodes"]) for k in list(prod) + list(cons)}
        prods = sorted(prod, key=lambda k: first[k], reverse=True)
        conss = sorted(cons, key=lambda k: first[k])
        if len(prod) >= 2:
            return prods + conss
        if len(prod) == 1 and self._shrinks(info, own):
            return prods + conss
        if len(cons) == 1:
            return conss + prods
        if prod:
            return prods + conss
        return conss

    @staticmethod
    def _shrinks(info, own):
        inb = 0
        for n in info["nodes"]:
            for a in n.all_input_nodes:
                if a in own:
                    continue
                for sh in _shapes(a):
                    inb = max(inb, _numel(sh, 1))
        outb = 0
        for n in info["nodes"]:
            if any(u not in own for u in n.users) or not n.users:
                for sh in _shapes(n):
                    outb = max(outb, _numel(sh, 1))
        return outb and inb and outb < inb

    def _absorb(self, groups, order):
        if not self.absorb:
            return groups
        self._heavy_seen = any(_is_heavy(n) for info in groups.values()
                               for n in info["nodes"])
        if not self._heavy_seen:
            print("WARNING: no heavy (matmul/attention) op recognized in graph; "
                  "falling back to weight-rank heuristic for stage absorption")
        for _ in range(8):
            containers = self._containers(groups)
            home = {n: k for k, info in groups.items() for n in info["nodes"]}
            succ = self._group_graph(groups, home)
            moved = False
            for key in list(groups.keys()):
                if len(groups) <= 1 or key not in groups:
                    continue
                info = groups[key]
                if not self._transparent(key, info, order, containers):
                    continue
                own = set(info["nodes"])
                cons, prod = OrderedDict(), OrderedDict()
                for n in info["nodes"]:
                    for u in n.users:
                        k = home.get(u)
                        if k is not None and k != key:
                            cons[k] = None
                    for a in n.all_input_nodes:
                        if self._param(a):
                            continue
                        k = home.get(a)
                        if k is not None and k != key:
                            prod[k] = None
                tgt = None
                for t in self._targets(info, own, prod, cons, groups, order):
                    if t in groups and t != key and not self._creates_cycle(succ, key, t):
                        tgt = t
                        break
                if tgt is None:
                    continue
                groups[tgt]["nodes"].extend(info["nodes"])
                groups[tgt]["keys"].update(info.get("keys", {}))
                groups[tgt].setdefault("absorbed", []).append(key)
                for n in info["nodes"]:
                    home[n] = tgt
                self._merge_succ(succ, key, tgt)
                del groups[key]
                moved = True
            if not moved:
                break
        return groups

    def _topo_order(self, by_key, order):
        home = {n: k for k, info in by_key.items() for n in info["nodes"]}
        succ = self._group_graph(by_key, home)
        first = {k: min(order[n] for n in info["nodes"]) for k, info in by_key.items()}
        indeg = {k: 0 for k in by_key}
        for k, out in succ.items():
            for o in out:
                indeg[o] += 1
        heap = [(first[k], k) for k, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        out = []
        while heap:
            _, k = heapq.heappop(heap)
            out.append(k)
            for o in succ[k]:
                indeg[o] -= 1
                if indeg[o] == 0:
                    heapq.heappush(heap, (first[o], o))
        if len(out) < len(by_key):
            stuck = [k for k in by_key if indeg[k] > 0]
            raise RuntimeError(f"cycle between stage groups: {stuck[:8]}")
        return out

    def _materialize(self, by_key, order):
        for i, key in enumerate(self._topo_order(by_key, order)):
            info = by_key[key]
            info["nodes"].sort(key=lambda n: order[n])
            base = self._base(key)
            st = Stage(
                name=key,
                role=self._role(base, info["cls"]),
                cls=info["cls"],
                candidates=info["devs"] or [self.default_device],
                nodes=info["nodes"],
                index=i,
                members=list(info.get("keys", {}) or [base]),
            )
            self.stages.append(st)
            for n in info["nodes"]:
                self.node_stage[n] = i

        bases = [self._base(st.name) for st in self.stages]
        for st in self.stages:
            b = self._base(st.name)
            st.is_parent_module = any(o != b and o.startswith(b + ".") for o in bases)
        for st in self.stages:
            self._finalize(st)

    def _finalize(self, st: Stage):
        nodeset = set(st.nodes)
        seen = set()
        edges = OrderedDict()
        pbytes = 0
        for n in st.nodes:
            for a in n.all_input_nodes:
                if a in nodeset or id(a) in seen:
                    continue
                seen.add(id(a))
                if self._param(a):
                    for t in _tensors(a):
                        sn = _static_numel(tuple(_concrete(d) for d in t.shape))
                        if sn:
                            pbytes += sn * _elem_size(t, self.bpe)
                    st.weight_shapes.extend(_shapes(a))
                    continue
                j = self.node_stage.get(a)
                if j is not None and j >= st.index:
                    raise RuntimeError(f"back edge {self.stages[j].name} -> {st.name} "
                                       f"via node {a.name}")
                edges.setdefault(j, []).extend(_shapes(a))
        st.param_bytes = pbytes
        st.inputs = [StageEdge(p, sh) for p, sh in edges.items()]
        st.predecessors = [p for p in edges.keys() if p is not None]

        outs = []
        for n in st.nodes:
            users = list(n.users)
            if not users or any(u not in nodeset for u in users):
                outs.extend(_shapes(n))
        if not outs and st.nodes:
            outs = _shapes(st.nodes[-1])
        st.out_shapes = outs


def transfer_fn(latency_table, startup_us, bytes_per_element=BYTES_PER_ELEMENT):
    def f(num_elements, dst, src):
        if dst == src:
            return 0.0
        bw = float(latency_table[dst][src]) / 1e6
        return (num_elements * bytes_per_element) / bw + startup_us
    return f


class DPPlanner:
    def __init__(self, stages: List[Stage], device_list: Dict[str, Any], init_device: str,
                 transfer: Callable,
                 budgets: Optional[Dict[str, int]] = None,
                 kv_bytes: Optional[Callable] = None, kv_roles=("attn",),
                 kv_filter: Optional[Callable] = None,
                 state_bytes: Optional[Callable] = None, state_roles=("mixer",),
                 dominance=True):
        self.stages = stages
        self.device_list = device_list
        self.devices = list(device_list.keys())
        if init_device not in self.devices:
            self.devices.append(init_device)
        self.didx = {d: k for k, d in enumerate(self.devices)}
        self.init_device = init_device
        self.transfer = transfer
        self.budgets = dict(budgets or {})
        self.kv_bytes = kv_bytes
        self.kv_roles = set(kv_roles or ())
        self.kv_filter = kv_filter
        self.state_bytes = state_bytes
        self.state_roles = set(state_roles or ())
        self.dominance = dominance

    def compute_latency(self, st, device, batch_size, seq_len, is_prefill):
        db = self.device_list.get(device)
        if db is None:
            return math.inf
        db = getattr(db, "database", db)
        return db.latency(stage=st, batch_size=batch_size,
                          seq_len=seq_len, is_prefill=is_prefill)

    def _has_kv(self, st):
        if st.is_parent_module or st.role not in self.kv_roles:
            return False
        return True if self.kv_filter is None else bool(self.kv_filter(st))

    def _bytes(self, st, batch_size, seq_len):
        b = st.param_bytes
        if self.kv_bytes is not None and self._has_kv(st):
            b += self.kv_bytes(st, batch_size, seq_len)
        if (self.state_bytes is not None and st.role in self.state_roles
                and not st.is_parent_module):
            b += self.state_bytes(st, batch_size, seq_len)
        return int(b)

    def _price_all(self, batch_size, seq_len, is_prefill):
        out = []
        for st in self.stages:
            comp = {}
            for d in st.candidates:
                c = self.compute_latency(st, d, batch_size, seq_len, is_prefill)
                if c == math.inf:
                    continue
                comp[d] = float(c)
            out.append(comp)
        return out

    def _precheck(self, stage_bytes):
        per = defaultdict(int)
        mandatory = 0
        for i, st in enumerate(self.stages):
            cands = st.candidates or []
            mem = [d for d in cands if d in self.budgets]
            if not cands or len(mem) != len(cands):
                continue
            mandatory += stage_bytes[i]
            if len(set(mem)) == 1:
                per[mem[0]] += stage_bytes[i]
        if any(b > self.budgets[d] for d, b in per.items()):
            print("Error mandatory per device")
            return False
        cap = sum(self.budgets[d] for d in self.devices if d in self.budgets)
        if mandatory > cap:
            print(f"cannot fit {mandatory/1e9} GB in:")
            for d in self.devices:
                if d in self.budgets:
                    print(f"{d} --> {self.budgets[d]/1e9} GB")
            return False
        return True

    def _buckets(self, stage_bytes, tracked):
        g = 0
        for b in stage_bytes:
            if b > 0:
                g = gcd(g, int(b))
        bucket = g or 1
        units = [int(b) // bucket if b > 0 else 0 for b in stage_bytes]
        budget_units = [int(self.budgets[d] // bucket) for d in tracked]
        return units, budget_units

    @staticmethod
    def _release_times(i, preds, dev_of, own_fin):
        rel = {}
        for p in preds:
            pdev = dev_of[p]
            k = p
            while k + 1 < i and dev_of.get(k + 1) == pdev:
                k += 1
            rel[p] = own_fin[k]
        return rel

    def _arrival(self, st, d, dev_of, avail, release, seed, toks, xcache):
        di = self.didx[d]
        arrival, critical = 0.0, None
        links = defaultdict(list)
        for e in st.inputs:
            pl = e.payload(toks)
            if pl <= 0:
                continue
            if e.producer is None:
                if d != self.init_device:
                    links[self.init_device].append(
                        (0.0, self.transfer(pl, d, self.init_device), None))
                continue
            p = e.producer
            have = avail[p][di]
            if have < math.inf:
                if have > arrival:
                    arrival, critical = have, p
                continue
            pdev = dev_of[p]
            key = (p, pdev, d)
            x = xcache.get(key)
            if x is None:
                x = self.transfer(pl, d, pdev)
                xcache[key] = x
            links[pdev].append((release[p], x, p))
        copies = {}
        for items in links.values():
            items.sort(key=lambda t: t[0])
            free = seed
            for ready, x, p in items:
                free = (ready if ready > free else free) + x
                if p is not None:
                    copies[p] = free
                if free > arrival:
                    arrival, critical = free, p
        return arrival, critical, copies

    @staticmethod
    def _pareto(cands, eps=1e-9):
        cands = sorted(cands, key=lambda c: sum(c[0]) + c[1])
        kept = []
        for used, g, ffin, node in cands:
            dominated = False
            for ku, kg, kf, _ in kept:
                if (kg <= g + eps
                        and all(a <= b for a, b in zip(ku, used))
                        and all(a <= b + eps
                                for ta, tb in zip(kf, ffin)
                                for a, b in zip(ta, tb))):
                    dominated = True
                    break
            if not dominated:
                kept.append((used, g, ffin, node))
        return kept

    def run(self, batch_size, seq_len, is_prefill, verbose=False):
        n = len(self.stages)
        if n == 0:
            return 0.0, {}
        toks = batch_size * seq_len if is_prefill else batch_size
        for st in self.stages:
            st.attach_dims(toks)

        stage_bytes = [self._bytes(st, batch_size, seq_len) for st in self.stages]
        if not self._precheck(stage_bytes):
            return math.inf, {}
        comp_all = self._price_all(batch_size, seq_len, is_prefill)

        tracked = [d for d in self.devices if d in self.budgets]
        worst = {d: 0 for d in tracked}
        for i, st in enumerate(self.stages):
            for d in (st.candidates or self.devices):
                if d in worst:
                    worst[d] += stage_bytes[i]
        tracked = [d for d in tracked if worst[d] > self.budgets[d]]
        midx = {d: k for k, d in enumerate(tracked)}
        M = len(tracked)
        units, budget_units = self._buckets(stage_bytes, tracked)

        last_use = list(range(n))
        for i, st in enumerate(self.stages):
            for p in st.predecessors:
                if i > last_use[p]:
                    last_use[p] = i

        nd = len(self.devices)
        states = {(): [(tuple(0 for _ in range(M)), 0.0, (), None)]}
        for i, st in enumerate(self.stages):
            preds = st.predecessors
            lo = min((k for k in range(i + 1) if last_use[k] > i), default=i)
            keep = range(lo, i + 1)
            comp = comp_all[i]
            su = units[i]
            xcache = {}
            new = defaultdict(list)
            for fk, group in states.items():
                pos = {k: j for j, (k, _) in enumerate(fk)}
                dev_of = dict(fk)
                prev = i - 1
                prev_dev = dev_of.get(prev)
                for used, gmax, ffin, node in group:
                    own_fin = {k: ffin[pos[k]][self.didx[dev_of[k]]] for k in pos}
                    release = self._release_times(i, preds, dev_of, own_fin)
                    avail = {p: ffin[pos[p]] for p in preds}
                    prev_start = (own_fin[prev] - comp_all[prev][prev_dev]
                                  if prev_dev is not None else 0.0)
                    for d, c in comp.items():
                        nused = used
                        di = midx.get(d)
                        if di is not None and su:
                            if used[di] + su > budget_units[di]:
                                continue
                            lst = list(used)
                            lst[di] += su
                            nused = tuple(lst)
                        seed = prev_start if prev_dev == d else 0.0
                        arrival, _, copies = self._arrival(st, d, dev_of, avail, release,
                                                           seed, toks, xcache)
                        fin = arrival + c
                        ngmax = gmax if gmax > fin else fin
                        ndi = self.didx[d]
                        nfk = tuple((k, d if k == i else dev_of[k]) for k in keep)
                        nffin = []
                        for k in keep:
                            if k == i:
                                nffin.append(tuple(fin if j == ndi else math.inf
                                                   for j in range(nd)))
                            elif k in copies:
                                t = list(ffin[pos[k]])
                                t[ndi] = copies[k]
                                nffin.append(tuple(t))
                            else:
                                nffin.append(ffin[pos[k]])
                        new[nfk].append((nused, ngmax, tuple(nffin), (i, d, node)))
            if not new:
                return math.inf, {}
            if self.dominance:
                states = {fk: self._pareto(g) for fk, g in new.items()}
            else:
                states = dict(new)
            if verbose:
                total = sum(len(g) for g in states.values())
                largest = max(len(g) for g in states.values())
                print(f"stage {i:4d} {st.name:<45} keys={len(states)} "
                      f"states={total} max_pareto={largest}")

        best = None
        for group in states.values():
            for cand in group:
                if best is None or cand[1] < best[1]:
                    best = cand
        if best is None:
            return math.inf, {}
        assignment = {}
        node = best[3]
        while node is not None:
            k, d, node = node
            assignment[k] = d
        return best[1], assignment

from remix_backend.database import PerfDatabase, TRACEDatabase, AttaccDatabase
import yaml
from functools import partial
MEM_BUDGETS = {"a100": int(0.95*80*1024**3),
                "h100": int(0.95*80*1024**3),
                "dram-pim": int(0.95*80*1024**3),
                "b200": int(0.95*192*1024**3),
                }
class GraphPartitionPlanner:
    def __init__(self, graph, device_map, available_devices, default_device, hf_config,
                 transfer_latencies_path, role_map=None,
                 bytes_per_element=BYTES_PER_ELEMENT,
                 kv_bytes=None, kv_roles=("attn",), kv_filter=None, config=None,
                 kv_dtype_bytes=None, state_bytes=None, state_roles=("mixer",),
                 absorb=True, priced_roles=None,
                 static_placeholders=True, glue_bytes=1 << 20, dominance=True,
                 ):
        bandiwtdhs =yaml.safe_load(open(transfer_latencies_path))
        startup_us = float(bandiwtdhs['startup_latency'])*1e6
        latency_table = bandiwtdhs['transfer_latencies']
                                
        init_device = default_device
        if len(available_devices) < 1:
            raise ValueError("Cannot find a partition without available_devices")
        DEVICE_SUPPORT_MAP = {
            "a100": partial(PerfDatabase, 'remix_backend/data/a100_database', hf_config),
            "h100": partial(PerfDatabase, 'remix_backend/data/h100_database', hf_config),
            "b200": partial(PerfDatabase, 'remix_backend/data/b200_database', hf_config),
            "3dcim": partial(TRACEDatabase, 'remix_backend/data/3dcim_database', hf_config),
            "dram-pim": partial(AttaccDatabase, 'remix_backend/data/attacc_database', hf_config)
        }
        device_list = {}
        for dev in available_devices:
            #device_list = {dev: DEVICE_SUPPORT_MAP[dev]() for dev in available_devices}
            try:
                device_backend = DEVICE_SUPPORT_MAP[dev]()
            except Exception as e:
                print(f"Database for device {dev} raised {e}")
                raise 
            device_list[dev] = device_backend
        budgets = {dev: MEM_BUDGETS[dev] for dev in MEM_BUDGETS.keys() if dev in available_devices}
        self.builder = DAG(graph, device_map, default_device,
                           role_map=role_map,
                           bytes_per_element=bytes_per_element,
                           absorb=absorb, priced_roles=priced_roles,
                           static_placeholders=static_placeholders,
                           glue_bytes=glue_bytes)
        self.stages = self.builder.stages
        self.default_device = default_device
        self.config = config
        if config is not None:
            kvb = kv_dtype_bytes or bytes_per_element
            if kv_bytes is None:
                kv_bytes = lambda st, bs, sl: kv_bytes_per_layer(config, bs, sl, kvb)
            if kv_filter is None:
                kv_filter = make_kv_filter(config)
            if state_bytes is None:
                state_bytes = lambda st, bs, sl: ssm_state_bytes(config, bs, bytes_per_element)
        self.dp = DPPlanner(self.stages, device_list, init_device,
                            transfer_fn(latency_table, bytes_per_element=bytes_per_element, startup_us=startup_us),
                            budgets=budgets,
                            kv_bytes=kv_bytes, kv_roles=kv_roles, kv_filter=kv_filter,
                            state_bytes=state_bytes, state_roles=state_roles,
                            dominance=dominance)

    def plan(self, batch_size, seq_len, is_prefill=False, verbose=False,
             prune_containers=True, raise_on_infeasible=True):
        cost, assignment = self.dp.run(batch_size, seq_len, is_prefill, verbose=verbose)
        if cost == math.inf or not assignment:
            msg = ("no feasible placement: memory budgets cannot be met "
                   "(check budgets, kv_bytes, and candidate device maps)")
            if raise_on_infeasible:
                raise RuntimeError(msg)
            print("WARNING:", msg, "- falling back to default device")
        module_map = {}
        node_map = {}
        order = {}
        for i, st in enumerate(self.stages):
            d = assignment.get(i, self.default_device)
            for m in [st.name] + list(st.members):
                if not m or m == "__root__" or "#" in m:
                    continue
                module_map[m] = d
                order.setdefault(m, i)
            for n in st.nodes:
                node_map[n.name] = d
        if prune_containers:
            module_map = self._prune_containers(module_map)
        module_map = dict(sorted(module_map.items(), key=lambda kv: order[kv[0]]))
        return cost, module_map, node_map

    def _prune_containers(self, module_map):
        keep = {st.name for st in self.stages}
        prefixes = set()
        for k in module_map:
            parts = k.split(".")
            for i in range(1, len(parts)):
                prefixes.add(".".join(parts[:i]))
        return {k: v for k, v in module_map.items()
                if k in keep or k not in prefixes}

    def dump(self, path, module_map, node_map=None, cost=None, meta=None):
        payload = {"partition_map": dict(module_map)}
        if cost is not None:
            payload["total_cost_us"] = None if cost == math.inf else float(cost)
        if meta:
            payload["meta"] = dict(meta)
        if node_map is not None:
            payload["node_map"] = dict(node_map)
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        if path.endswith((".yaml", ".yml")):
            import yaml
            with open(path, "w") as f:
                yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        else:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        print(f"wrote {path} ({len(payload['partition_map'])} modules)")
        return path

    @staticmethod
    def load(path):
        if path.endswith((".yaml", ".yml")):
            import yaml
            with open(path) as f:
                payload = yaml.safe_load(f)
        else:
            with open(path) as f:
                payload = json.load(f)
        return payload["partition_map"]

    def memory_report(self, batch_size, seq_len, is_prefill):
        dp = self.dp
        sb = [dp._bytes(st, batch_size, seq_len) for st in self.stages]
        worst = {}
        for i, st in enumerate(self.stages):
            for d in st.candidates:
                worst[d] = worst.get(d, 0) + sb[i]
        print(f"stages={len(self.stages)} total_bytes={sum(sb)/1e9:.2f} GB")
        for d, w in worst.items():
            b = dp.budgets.get(d)
            tag = "UNTRACKED" if b is None else ("tracked" if w > b else "fits")
            bs = "-" if b is None else f"{b/1e9:.2f}"
            print(f"  {d:<10} worst={w/1e9:8.2f} GB  budget={bs:>8} GB  {tag}")
        cont = [st.name for st in self.stages if st.is_parent_module]
        if cont:
            print(f"  {len(cont)} container stages (no KV charged), e.g. {cont[:3]}")
        kvn = [st.name for st in self.stages if dp._has_kv(st)]
        print(f"  {len(kvn)} stages charged KV")
        ssm = [st.name for st in self.stages
               if dp.state_bytes is not None and st.role in dp.state_roles
               and not st.is_parent_module]
        print(f"  {len(ssm)} stages charged SSM state")
        zero = [st.name for i, st in enumerate(self.stages) if sb[i] == 0]
        if zero:
            print(f"  {len(zero)} stages with 0 bytes, e.g. {zero[:4]}")
        return sb

    def edge_report(self, toks=1, limit=12):
        multi = [st for st in self.stages if len(st.inputs) > 1]
        print(f"{len(multi)}/{len(self.stages)} stages with >1 input edge")
        for st in self.stages[:limit]:
            srcs = [(self.stages[e.producer].name if e.producer is not None else "ENTRY",
                     e.shapes, e.payload(toks)) for e in st.inputs]
            print(f"  {st.name:<40} in={srcs}")
        return multi

    def node_report(self, match, limit=20):
        hit = 0
        for st in self.stages:
            for n in st.nodes:
                t = str(getattr(n, "target", "")) + " " + n.name
                if match in t:
                    print(f"  {n.name:<24} target={getattr(n,'target',None)} "
                          f"stage={st.name} key={module_key(n)} "
                          f"in={[a.name for a in n.all_input_nodes]} "
                          f"users={[u.name for u in n.users]}")
                    hit += 1
                    if hit >= limit:
                        return
        if not hit:
            print(f"  no node matching {match!r}")

    def cost_report(self, batch_size, seq_len, is_prefill):
        dp = self.dp
        feas = defaultdict(lambda: defaultdict(int))
        forced = defaultdict(int)
        for st in self.stages:
            ok = []
            for d in st.candidates:
                c = dp.compute_latency(st, d, batch_size, seq_len, is_prefill)
                if c != math.inf:
                    ok.append(d)
                    feas[st.role][d] += 1
            if not ok:
                forced[st.role] += 1
        for role in sorted(set(list(feas) + list(forced))):
            devs = ", ".join(f"{d}:{n}" for d, n in feas[role].items()) or "NONE"
            tag = f"  !! {forced[role]} stages with no priced device" if forced[role] else ""
            print(f"  {role:<16} {devs}{tag}")
        return feas, forced

    def feasibility_report(self, batch_size, seq_len, is_prefill):
        dp = self.dp
        sb = [dp._bytes(st, batch_size, seq_len) for st in self.stages]
        comp_all = dp._price_all(batch_size, seq_len, is_prefill)
        floor = defaultdict(int)
        for i, comp in enumerate(comp_all):
            if len(comp) == 1:
                floor[next(iter(comp))] += sb[i]
        for d, b in floor.items():
            bud = dp.budgets.get(d)
            tag = "" if bud is None else ("  OVER BUDGET" if b > bud else "")
            bs = "-" if bud is None else f"{bud/1e9:.2f}"
            print(f"  {d:<10} forced={b/1e9:8.2f} GB  budget={bs:>8} GB{tag}")
        dp._precheck(sb)
        return floor

    def breakdown(self, batch_size, seq_len, is_prefill, assignment):
        toks = batch_size * seq_len if is_prefill else batch_size
        rows = []
        for i, st in enumerate(self.stages):
            st.attach_dims(toks)
            d = assignment.get(st.name, self.default_device)
            rows.append({
                "name": st.name,
                "role": st.role,
                "device": d,
                "compute_us": self.dp.compute_latency(st, d, batch_size, seq_len, is_prefill),
                "out_elems": st.out_elems,
                "param_mb": st.param_bytes / 1e6,
                "mem_mb": self.dp._bytes(st, batch_size, seq_len) / 1e6,
                "preds": [self.stages[p].name for p in st.predecessors],
                "modules": st.members,
            })
        return rows