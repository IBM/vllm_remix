#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from .base_backend import BaseBackend
import torch
from contextlib import nullcontext
import yaml 

class DeviceTransferBackend(BaseBackend):
    
    def __init__(self, target_device, source_device, transfer_latencies_path):
        assert target_device is not None
        self.target_device = target_device
        with open(transfer_latencies_path) as f:
            data = yaml.safe_load(f)
        self.startup_latency = float(data['startup_latency'])
        data = data['transfer_latencies'] 
        if source_device.name in data.keys() and target_device.name in data[source_device.name].keys():
            self.bw = int(float(data[source_device.name][target_device.name]))
        else:
            self.bw = 64e9 # PCIe by default
        self.total_bytes = 0
        self.record_shapes = False
        self.name = 'transfer'
        print(f"Initializing transfer from {source_device} to {target_device}")

    def backend(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ):
        return graph_module

    def move_to_device(self, x, count=True):
        if isinstance(x, torch.Tensor):
            if count and self.record_shapes and not isinstance(x, torch.nn.Parameter):
                self.total_bytes += int(x.nelement()) * int(x.element_size())
            return self.target_device.backend.move_to_device(x)
        if isinstance(x, (list, tuple)):
            return type(x)(self.move_to_device(item, count) for item in x)
        if isinstance(x, dict):
            return {k: self.move_to_device(v, count) for k, v in x.items()}
        return x

    def get_profile_ctx(self):
        self.record_shapes = True
        return nullcontext()

    def return_latency(self, profiler, meta_ctx) -> float:
        timing =self.total_bytes / self.bw # PCIe
        #implement transfer latency here
        self.total_bytes = 0
        return timing + self.startup_latency
