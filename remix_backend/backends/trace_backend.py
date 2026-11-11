#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
import json
import yaml
from functools import partial
import torch
from .base_backend import BaseBackend
from tads.torch_compile_backend.compile_options import Level
from tads.torch_compile_backend.register_backend import TADSCompile as TRACECompile
from tads.torch_compile_backend.compile_options import CompileOptions
# Base cleaning/common patterns from the TADS framework
#from tads.torch_compile_backend.torch_fx_patterns import cleaning_patterns, common_patterns
from tads.ops.torch_fx_patterns import cleaning_patterns, common_patterns
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import aot_module
# Llama graph patterns — requires PYTHONPATH to include the TADS repo root
# (TADS/examples/__init__.py and TADS/examples/Llama/__init__.py must exist)
from remix_backend.trace.patterns_vllm import (
    AliasToInput, VllmRMSNorm, FuseKQV, FuseGM, SiluFP32Cast,
    PositionalEncodings, AttentionMasks, AttentionBlocks,
    IgnoreMulinRoPE, IgnoreSymAdd, IgnoreLMHead,
    LastBambaDense, StageDense, GroupedMMPattern, GroupedMMMul
)

from torch._functorch.aot_autograd import aot_function
from torch._decomp import core_aten_decompositions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cim_functions"))
import os
import tads.logging.logging as logging
from torch import device

# op = torch.ops.vllm.moe_forward.default
@torch.library.impl("vllm::moe_forward", "Functionalize")
def my_op_functionalize(*args, **kwargs):
    # activation, div, top_k   =args[:3]
    batch_args = args[0].size()[0]
    _to_copy_7 = torch.ops.aten._to_copy.default(args[1], dtype = torch.float32)
    _softmax = torch.ops.aten._softmax.default(_to_copy_7, -1, False)
    top_k = torch.ops.aten.topk.default(_softmax, 2)
    getitem_4 = top_k[0]
    getitem_5 = top_k[1]
    sum_1 = torch.ops.aten.sum.dim_IntList(getitem_4, [-1], True)
    div = torch.ops.aten.div.Tensor(getitem_4, sum_1)
    arange_4 = torch.ops.aten.arange.default(batch_args, device = args[0].device, pin_memory = False)
    unsqueeze_13 = torch.ops.aten.unsqueeze.default(arange_4, 1)
    expand_7 = torch.ops.aten.expand.default(unsqueeze_13, [-1, 2])
    mul_290 = batch_args * 2
    clone_2 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format)
    _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_2, [mul_290])
    view_14 = torch.ops.aten.view.default(getitem_5, [-1])
    sort = torch.ops.aten.sort.default(view_14)
    getitem_7 = sort[1]
    empty_like = torch.ops.aten.empty_like.default(getitem_7, pin_memory = False)
    arange_5 = torch.ops.aten.arange.default(mul_290, device =args[0].device, pin_memory = False)
    index_put = torch.ops.aten.index_put.default(empty_like, [getitem_7], arange_5)
    index = torch.ops.aten.index.Tensor(view_14, [getitem_7])
    view_15 = torch.ops.aten.view.default(div, [-1])
    index_1 = torch.ops.aten.index.Tensor(view_15, [getitem_7])
    index_2 = torch.ops.aten.index.Tensor(_unsafe_view_6, [getitem_7])
    index_3 = torch.ops.aten.index.Tensor(args[0], [index_2])
    _to_copy_8 = torch.ops.aten._to_copy.default(index, dtype = torch.int32)
    histc = torch.ops.aten.histc.default(_to_copy_8, 8, 0, 7)
    cumsum = torch.ops.aten.cumsum.default(histc, 0, dtype = torch.int32)
    arg15_1 = torch.ops.aten.new_ones.default(args[0],[8, 28672, 4096])
    transpose_5 = torch.ops.aten.transpose.int(arg15_1, -2, -1)
    _grouped_mm = torch.ops.aten._grouped_mm.default(index_3, transpose_5, cumsum)
    split = torch.ops.aten.split.Tensor(_grouped_mm, 14336, -1)
    getitem_8 = split[0]
    getitem_9 = split[1]
    silu = torch.ops.aten.silu.default(getitem_8)
    mul_330 = torch.ops.aten.mul.Tensor(silu, getitem_9)
    arg16_1 = torch.ops.aten.new_ones.default(args[0], [8, 4096, 14336])
    transpose_6 = torch.ops.aten.transpose.int(arg16_1, -2, -1)
    _grouped_mm_1 = torch.ops.aten._grouped_mm.default(mul_330, transpose_6, cumsum)
    unsqueeze_14 = torch.ops.aten.unsqueeze.default(index_1, -1)
    mul_336 = torch.ops.aten.mul.Tensor(_grouped_mm_1, unsqueeze_14)
    index_4 = torch.ops.aten.index.Tensor(mul_336, [index_put])
    view_16 = torch.ops.aten.view.default(index_4, [batch_args, 2, 4096])
    sum_2 = torch.ops.aten.sum.dim_IntList(view_16, [1])
    return sum_2
#return torch.ops.vllm.transformers_moe_forward(*args, **kwargs)


class TRACEBackend(BaseBackend):
    def __init__(self, compile_options: CompileOptions, vocab_size: int, timing_path: str = "timings.yaml"):
        compile_options.torch_patterns = [
            *cleaning_patterns,
            *common_patterns,
            AliasToInput,
            VllmRMSNorm,
            FuseKQV,
            SiluFP32Cast,
            FuseGM,
            StageDense,
            LastBambaDense,
            GroupedMMPattern,
            GroupedMMMul,
            *PositionalEncodings,
            *AttentionMasks,
            *AttentionBlocks,
            IgnoreMulinRoPE,
            IgnoreSymAdd,
            partial(IgnoreLMHead, vocab_size=vocab_size),
        ]
        compile_options.performance_logging = True
        compile_options.performance_logging_path = timing_path
        
        compile_options._log_level = logging.TRACE
        self.compiler = TRACECompile(compile_options)
        self.compile_options = compile_options
        self.timing_path = timing_path
        self.clock_frequency = 1e9

    def backend(self, graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        def wrapper_compiler(gm, fake_inputs):
            return self.compiler_fn(gm, fake_inputs)


        for node in graph_module.graph.nodes:
                if node.op == 'output':
                    out = node.args[0]
                    # normalize to a tuple output for AOTAutograd, idempotently
                    if isinstance(out, torch.fx.Node):
                        node.args = ((out,),)
                    break
        
        compiled_f = aot_autograd(
            fw_compiler=wrapper_compiler,
            bw_compiler=None,
            #decompositions=core_aten_decompositions(),
            #enable_functionalization=False,

        ) (graph_module, example_inputs)


        returns_tuple = graph_module.meta.get("returns_tuple", False)
        def wrapped(*args):
            result = compiled_f(*args)
            if returns_tuple:
                return result
            if isinstance(result, (tuple, list)) and len(result) == 1:
                return result[0]
            #_report(result)
            #return result[0] if modified_output else result
            return result

        return wrapped
        return compiled_f
        

    def return_latency(self, profiler, meta_ctx) -> float:
        def _collect_active_cycles(node):
            if isinstance(node, dict):
                if "active_cycles" in node:
                    cycles = node["active_cycles"]
                    if cycles:
                        yield from (v for pair in cycles for v in pair)
                for v in node.values():
                    yield from _collect_active_cycles(v)
            elif isinstance(node, list):
                for item in node:
                    yield from _collect_active_cycles(item)

        with open(self.timing_path, "r") as f:
            data = yaml.safe_load(f)
        all_cycles = list(_collect_active_cycles(data))
        lat = float(max(all_cycles))/self.clock_frequency if all_cycles else 0.0
        return lat


    def compiler_fn(self, gm, example_inputs):
        before = torch.cuda.memory_allocated()
        compiled = self.compiler.compile_graph(gm, example_inputs)
        after = torch.cuda.memory_allocated()
        return compiled

    def move_to_device(self, x):
        #return x.cuda()
        return x