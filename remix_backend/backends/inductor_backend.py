#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import torch
from remix_backend.utils import FunctionWrapperModule
from .base_backend import BaseBackend
import os
import json
import copy
from torch._inductor.compile_fx import compile_fx
from torch.profiler import profile, schedule

class CUDAInductorBackend(BaseBackend):
    """Backend that compiles for GPU via Inductor"""

    def __init__(self, profiler_config= None):
        if not os.path.exists(profiler_config.chrome_trace_path):
            os.makedirs(os.path.dirname(profiler_config.chrome_trace_path), exist_ok=True)
        self.timing_path = profiler_config.chrome_trace_path if str != "" else "traces/timings.json" 
        self.name='cuda'

    def get_path_to_perf(self) -> str:
        return self.timing_path

    def backend(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ):
        compiled_fn = compile_fx(graph_module, example_inputs)
        return compiled_fn

    def move_to_device(self, x):
        return x.cuda()


    def get_kernel_time(self, path: str, cat_lookup: str = "kernel"):
        with open(path, "r") as f:
            trace = json.load(f)
        kernels = {}
        for event in trace["traceEvents"]:
            cat = event.get("cat", "")
            name = event.get("name", "")
            dur = event.get("dur", 0)  # in microsecs
            if cat == cat_lookup:
                if name not in kernels:
                    kernels[name] = {"count": 0, "total_us": 0}
                kernels[name]["count"] += 1
                kernels[name]["total_us"] += dur
                continue
        total_time = 0
        for name, stats in sorted(kernels.items(), key=lambda x: -x[1]["total_us"]):
            total_time += stats["total_us"]
        return total_time

    def return_latency(self, profiler, meta_ctx) -> float:
        subgraph_name = meta_ctx.name
        profiler.export_chrome_trace(self.timing_path[:-5] + f"_{subgraph_name}_breakdown_latency.json")
        kernel_exec_time = self.get_kernel_time(
            path=self.timing_path[:-5] + f"_{subgraph_name}_breakdown_latency.json", cat_lookup="kernel"
        )
        launch_kernel_time = self.get_kernel_time(
            path=self.timing_path[:-5] + f"_{subgraph_name}_breakdown_latency.json",
            cat_lookup="cuda_runtime",
        )
        with open(self.timing_path[:-5] + f"_{subgraph_name}.json", "w") as f:
            json.dump(
                {
                    f"latency_{subgraph_name}": kernel_exec_time,
                    f"launch_time_{subgraph_name}": launch_kernel_time,
                },
                f,
            )
        # TODO: enable a flag to account for the launch time in the latency
        return kernel_exec_time * 1e-6/meta_ctx.ncalls

    def get_profile_ctx(self):
        return profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            )