#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
import json
import torch
import re
from loguru import logger
import os
from vllm.compilation import monitor




@dataclass
class PerformanceMetric:
    latency: int


class TimingCollector:
    """
    Collects timing information for all named submodules in the graph.
    """

    def __init__(self, path: str = ""):
        self.timings = {}
        self.is_active = False
        self.path = 'profiler/full_partition.json' if path == "" else path
        self.batch_path = self.path
        self.batch = 0
        self.pref_throughput = []
        self.pref_time = 0.0
        self.dec_throughput = []
        self.tpot = []
        self.ttft = -1
        self.computed_tokens = 0
        self.stored_sched = None


    def record(self, name, pm: PerformanceMetric):
        self.timings.setdefault(name, []).append(pm)

    def set_path(self, path: str):
        self.path = path

    def summary(self):
        if not self.is_active:
            return
        timings_dict = {
            k: sum([pm.latency for pm in v]) / len(v) for k, v in self.timings.items()
        }
        latency = sum(timings_dict.values())
        if self.pref_tokens > 0 and self.dec_tokens == 0:
            self.pref_throughput.append(self.pref_tokens / latency)
            self.pref_time += latency
        if self.dec_tokens > 0 and self.pref_tokens == 0:
            self.dec_throughput.append(self.dec_tokens / latency)
            self.tpot.append(latency/self.dec_tokens)
            if self.ttft == -1:
                self.ttft = self.pref_time
        logger.info(f"Input speed {sum(self.pref_throughput)/len(self.pref_throughput)} toks/s")
        logger.info(f"Out speed {sum(self.dec_throughput)/len(self.dec_throughput) if len(self.dec_throughput) !=0 else 0.0} toks/s")
        logger.info(f"Current input speed {self.pref_throughput[-1]}")
        logger.info(f"Current output speed {self.dec_throughput[-1] if len(self.dec_throughput)!=0 else 0}")
        self.create_chrome_trace(timings_dict)
        with open(self.path, 'a') as f:
            json.dump(
                {f"total_latency_{self.batch}": latency},
                f
            )
            f.write('\n')
        return timings_dict

    def start(self):
        self.is_active = True

    def stop(self):
        self.is_active = False

    def set_timings_path(self, name: str):
        if not self.is_active:
            return
        toks = re.findall(r"\((\d+)\)", name)
        toks = [int(x) for x in toks] # prefill, decode tokens
        self.pref_tokens, self.dec_tokens = toks[0], toks[1]
        self.timings = {}
        self.batch_path = self.path[:-5] + f"{self.batch}_pref_{self.pref_tokens}_dec_{self.dec_tokens}" + self.path[-5:] if any(x != 0 for x in toks) else self.path[:-5] + '_end' + self.path[-5:]
        self.batch += 1

    def create_chrome_trace(self, timings_dict):
        trace_events = []
        pid = 1
        device_tids = {}
        tid_counter = 1
        
        device_colors = {
            'cuda': "olive",
            'a100': "olive",
            'b200': "olive",
            'h100': "olive",
            '3dcim': 'rail_response',
            'dram_pin': 'blue',
            'transfer': "generic_work"
        }
        
        current_time = 0

        for key, latency in timings_dict.items():
            submod_marker = '_submod_'
            idx = key.index(submod_marker)
            prefix = key[:idx]
            submod_num = key[idx + len(submod_marker):]

            parts = prefix.split('_', 1)
            device = parts[0]
            kernels = parts[1].split('-') if len(parts) > 1 else []

            if device not in device_tids:
                device_tids[device] = tid_counter
                tid_counter += 1

            tid = device_tids[device]
            latency_us = latency * 1e6

            event = {
                'name': f'{device}_{submod_num}',
                'cat': device,
                'ph': 'X',
                'ts': current_time,
                'dur': latency_us,
                'pid': pid,
                'tid': tid,
                'args': {
                    'latency_s': latency,
                    'submodule': submod_num,
                    'kernels': kernels,
                    'kernel_count': len(kernels),
                }
            }

            if device == 'transfer':
                event['args']['style'] = 'dashed'

            if device in device_colors:
                event['cname'] = device_colors[device]

            trace_events.append(event)
            current_time += latency_us

        for device, tid in device_tids.items():
            trace_events.append({
                'name': 'thread_name',
                'ph': 'M',
                'pid': pid,
                'tid': tid,
                'args': {'name': device}
            })
        avg_input_speed = sum(self.pref_throughput) / len(self.pref_throughput) if self.pref_throughput else 0.0
        avg_out_speed   = (sum(self.dec_throughput) / len(self.dec_throughput)
                        if len(self.dec_throughput) != 0 else 0.0)
        trace_events.append({
            'name': 'throughput_stats',
            'ph': 'M',
            'pid': pid,
            'tid': 0,
            'args': {
                'input_speed_toks_s': avg_input_speed,
                'out_speed_toks_s':   avg_out_speed,
            }
        })

        trace_data = {'traceEvents': trace_events}

        import tempfile

        dir_name = os.path.dirname(self.batch_path)

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=dir_name,
            delete=False
        ) as tmp:
            json.dump(trace_data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name

        os.replace(tmp_path, self.batch_path)

@dataclass
class MetaCtx:
    name: str = ""
    ncalls: int = 1
    dec_toks: int = 1
    pref_toks: int = 1
    scheduler_output: object = None

class TimeWrapper(torch.nn.Module):
    """
    Wraps a function module wrapper to aggregate timing information.
    """

    def __init__(
        self, mod: torch.nn.Module, aggregator: TimingCollector, target_backend: str, name: str=""
    ):
        super().__init__()
        self.mod = mod
        self.kernel_name = "transfer" # transfer
        if hasattr(mod, "meta"): # attention not wrapped in PieceWiseBakcned, mod is directly GraphModule
            self.kernel_name = mod.meta.get("module_components", "")
        elif hasattr(mod, "graph"): # rest of modules
            self.kernel_name = mod.graph.meta.get("module_components", "")
        self.target_backend = target_backend
        self.aggregator = aggregator
        self.collector = aggregator.collector
        self.latency = None
        self.name = f"{self.target_backend.name}_{self.kernel_name}_submod_{name.split('_')[-1]}"

    def record_times(self, profile_ctx, reps=1):
        #print("read:", id(self.aggregator), self.aggregator.scheduler_output)
        self.latency = self.target_backend.return_latency(profile_ctx, MetaCtx(self.name,
                                                                  reps,
                                                                  getattr(self.collector, "dec_tokens", 0),
                                                                  getattr(self.collector, "pref_tokens", 1),
                                                                  scheduler_output=getattr(self.aggregator, "scheduler_output", None)))
        self.collector.record(self.name, PerformanceMetric(self.latency))
        return

    def get_backend_name(self):
        return self.mod.target_backend.name

    def forward(self, *args, **kwargs):

        capturing = (
        monitor.cudagraph_capturing_enabled
        or torch.cuda.is_current_stream_capturing()
        )
        if capturing or not self.collector.is_active:
            out = self.mod(*args, **kwargs)
            return out
        with self.target_backend.get_profile_ctx() as profile_ctx:
            out = self.mod(*args, **kwargs)
            # FIXME for CPU profiling avoid calling sync
            if profile_ctx:
                torch.cuda.synchronize()
                profile_ctx.step()
        self.record_times(profile_ctx)
        return out

