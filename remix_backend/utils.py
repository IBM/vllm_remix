#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch._ops import OpOverload, OpOverloadPacket


class FunctionWrapperModule(torch.nn.Module):
    def __init__(self, fn, target_backend, default_backend=None):
        super().__init__()
        self.fn = fn
        self.target_backend = target_backend
        self.default_backend = default_backend
        self.latency = 0
    
    def get_backend_name(self):
        return self.target_backend.name
    
    def forward(self, *args, **kwargs):
        # TODO Use the wrapper to get latencies for alternaive backends exclusively
        # rename it as ProfileWrapper as the forward call does nott have to be intercepted
        result = self.fn(*args, **kwargs)
        return result

def curate_shape(shape):
    def get_real_dim(dim):
        if isinstance(dim, int):
            return dim
        elif isinstance(dim, torch.SymInt):
            return dim.node.hint
    # Avoid returning torch.Size with SymInt
    return torch.Size([get_real_dim(v) for v in shape])

def get_node_shape(node):
    if "val" in node.meta:
        if hasattr(node.meta["val"], "shape"):
            return curate_shape(node.meta["val"].shape)
        elif isinstance(node.meta["val"], (tuple, list)):
            if all( [hasattr(v, "shape") for v in node.meta["val"]]):
                return [curate_shape(v.shape) for v in node.meta["val"]]
            else:
                def get_val_shape(v):
                    if hasattr(v, "shape"):
                        return curate_shape(v.shape)
                    elif isinstance(v, torch.SymInt):
                        return v.node.hint if hasattr(v, 'node') else v
                    elif isinstance(v, int):
                        return v
                    return None
                return [get_val_shape(v) for v in node.meta["val"]]
        elif isinstance(node.meta["val"], torch.SymInt): # Dynamic shapes
            if hasattr(node.meta["val"], 'node'):
                return node.meta["val"].node.hint  # Sinle value
    elif "tensor_meta" in node.meta:
        if hasattr(node.meta["tensor_meta"], "shape"):
            return node.meta["tensor_meta"].shape
        elif isinstance(node.meta["tensor_meta"], (tuple, list)) and all(
            [hasattr(v, "shape") for v in node.meta["tensor_meta"]]
        ):
            return [v.shape for v in node.meta["tensor_meta"]]
        else:
            raise NotImplementedError
    else:
        return torch.Size([0])


def get_node_dtype(node):
    if "val" in node.meta and node.meta["val"] is not None:
        try:
            return node.meta["val"][0].dtype
        except:
            return torch.float16
    elif "tensor_meta" in node.meta:
        return node.meta["tensor_meta"].dtype
    else:
        return torch.float16


def gen_random_inputs(shape, dtype=torch.float32):
    concrete_shape = []
    if isinstance(shape, (int, float)):
        return shape
    if isinstance(shape, torch.SymInt):
        return shape.node.hint if hasattr(shape, 'node') else int(shape)

    for dim in shape:
        # check for Dynamic shapes
        if isinstance(dim, torch.SymInt):
            concrete_shape.append(dim.node.hint if hasattr(dim.node, 'hint') else 1)
        elif hasattr(dim, '__int__'):
            concrete_shape.append(int(dim))
        else:
            concrete_shape.append(dim)
    
    try:
        return torch.randn(concrete_shape, dtype=dtype)
    except:
        return torch.zeros(concrete_shape, dtype=dtype)

class TaggingPassWrapper(PassBase):
    """
    Wraps any pass to add metadata tagging after decomposition
    args:
        inner_pass: PassManager
        target_device: str (optional) - device name to tag new nodes with
    """

    def __init__(self, inner_pass, target_device: str = None):
        super().__init__()
        self.inner_pass = inner_pass
        self.target_device = target_device

    def call(self, graph_module) -> PassResult:
        nodes_before = set(node.name for node in graph_module.graph.nodes)
        result = self.inner_pass(graph_module)

        if isinstance(result, PassResult):
            graph_module = result.graph_module
        else:
            graph_module = result

        for node in graph_module.graph.nodes:
            if not isinstance(node.target, (OpOverload, OpOverloadPacket)):
                continue
            is_new = node.name not in nodes_before

            if is_new or not node.meta:
                if self.target_device:
                    node.meta["device"] = self.target_device

        return PassResult(graph_module, True)


def plot_graph(graph: torch.fx.GraphModule, file_path: str, label: str = ""):
    assert len(file_path) > 4 and (
        file_path[-4:] == ".svg" or file_path[-4:] == ".pdf"
    ), "Format must be svg"


def visualize_split_graph(split_gm, output_path='split_graph.svg', device_colors=None):
    import graphviz
    import operator

    if device_colors is None:
        device_colors = {
            'h100': '#66b3e3',
            'a100': '#66b3e3',
            'b200': '#66b3e3',
            '3dcim': '#ffab61',
            'dram-pim': "#a247a4",
            'transfer': '#979899',
        }

    fmt = output_path.rsplit('.', 1)[-1]
    dot = graphviz.Digraph(format=fmt)
    dot.attr(rankdir='TB', compound='true', nodesep='0.3', ranksep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='8')

    submod_nodes = {}
    for n in split_gm.graph.nodes:
        if n.op == 'call_module':
            submod_nodes[n.name] = n

    submod_order = list(submod_nodes.keys())

    for sname in submod_order:
        node = submod_nodes[sname]
        submod = getattr(split_gm, node.target, None)
        meta = getattr(submod, 'meta', {}) if submod is not None else {}
        device = meta.get('device', 'unknown')
        module_name = meta.get('module_name', '')

        device_key = device.name if device.name in device_colors else (
            'transfer' if 'transfer' in str(device).lower() else 'cuda'
        )
        color = device_colors.get(device_key, '#cccccc')

        with dot.subgraph(name=f'cluster_{sname}') as c:
            c.attr(
                label=f'{sname}\n[{device.name}]' + (f'\n{module_name}' if module_name else ''),
                style='filled,rounded',
                fillcolor=color + '40',
                color=color,
                penwidth='2',
                fontsize='11',
            )

            inner_nodes = []
            if submod is not None and hasattr(submod, 'graph'):
                for inner in submod.graph.nodes:
                    if inner.op in ('placeholder', 'output'):
                        continue
                    if inner.op == 'call_function' and inner.target == operator.getitem:
                        continue
                    label = _node_label(inner)
                    nid = f'{sname}__{inner.name}'
                    c.node(nid, label=label, shape='box', style='filled', fillcolor='white')
                    inner_nodes.append(nid)
            else:
                nid = f'{sname}__virtual'
                c.node(nid, label=f'{sname}\n(transfer)', shape='box',
                       style='filled,dashed', fillcolor='white')
                inner_nodes.append(nid)

            for a, b in zip(inner_nodes, inner_nodes[1:]):
                c.edge(a, b)

            submod_nodes[sname] = (node, inner_nodes)

    for i in range(len(submod_order) - 1):
        a_name = submod_order[i]
        b_name = submod_order[i + 1]
        _, a_inner = submod_nodes[a_name]
        _, b_inner = submod_nodes[b_name]
        if a_inner and b_inner:
            dot.edge(
                a_inner[-1], b_inner[0],
                ltail=f'cluster_{a_name}',
                lhead=f'cluster_{b_name}',
                penwidth='1.5',
            )

    out = dot.render(output_path.rsplit('.', 1)[0], cleanup=True)
    return out


def _node_label(node):
    if node.op == 'call_function':
        t = node.target
        name = getattr(t, '__name__', str(t))
    elif node.op == 'call_method':
        name = node.target
    elif node.op == 'call_module':
        name = node.target
    else:
        name = node.op
    return f'{name}'

import json
import glob
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_timeline(pattern, files, token_budget, ):
    runs = []
    for f in files:
        m = pattern.search(f)
        if m:
            runs.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), f))
    runs.sort()
    
    color_map = {'cuda': '#66b3e3', 'tads': '#ffab61', 'transfer': '#979899'}
    TOKEN_PREFILL_COLOR = "#B789CA"
    TOKEN_DECODE_COLOR  = "#98d746"
    TOKEN_EMPTY_COLOR   = "#e5e3e3"
    n_runs = len(runs)
    ROW_HEIGHT_PX  = 375
    HEADER_FRAC    = 0.25  # fraction of each subplot slot reserved for header
    INTER_SPACING  = 0.2  # gap between subplot slots in paper coords

    fig = make_subplots(rows=n_runs, cols=1, vertical_spacing=0.0)

    seen_devices = set()

    slot = 1.0 / n_runs  # paper height per run slot

    for idx, (n, m, k, filepath) in enumerate(runs):
        with open(filepath) as f:
            events = json.load(f)['traceEvents']
        throughput = next(
            (e['args'] for e in events
             if e.get('ph') == 'M' and e.get('name') == 'throughput_stats'),
            None
        )
        input_speed = throughput['input_speed_toks_s'] if throughput else None
        out_speed   = throughput['out_speed_toks_s']   if throughput else None

        row = idx + 1
        axis_idx = '' if row == 1 else str(row)

        # Manual domain: top portion of slot = header, rest = chart
        slot_top = 1.0 - idx * slot
        slot_bot = slot_top - slot
        chart_top = slot_top - HEADER_FRAC * slot
        chart_bot = slot_bot + INTER_SPACING * slot
        header_y = (slot_top + chart_top) / 2
# Then inside the per-run annotation block, after the bold title annotation, add:

        if input_speed is not None:
            fig.add_annotation(
                xref='paper', yref='paper',
                x=0.99, y=header_y + 0.012,
                text=f'<b>in:</b> {input_speed:.1f} tok/s  <b>out:</b> {out_speed:.1f} tok/s',
                showarrow=False, xanchor='right',
                font=dict(size=11, color='#333'),
            )

        rows = []
        for e in events:
            if e.get('ph') != 'X':
                continue
            kernels = e['args'].get('kernels', [])
            rows.append({
                'device': e['cat'],
                'name': e['name'],
                'start_us': e['ts'],
                'duration_us': e['dur'],
                'kernels': '<br>'.join(kernels) if kernels else '(none)',
            })

        df = pd.DataFrame(rows)
        row = idx + 1
        axis_idx = '' if row == 1 else str(row)

        # Manual domain: top portion of slot = header, rest = chart
        slot_top = 1.0 - idx * slot
        slot_bot = slot_top - slot
        chart_top = slot_top - HEADER_FRAC * slot
        chart_bot = slot_bot + INTER_SPACING * slot

        fig.update_layout(**{
            f'yaxis{axis_idx}': dict(autorange='reversed', domain=[chart_bot, chart_top]),
            f'xaxis{axis_idx}': dict(title='Time (µs)', domain=[0.0, 1.0]),
        })
        for device in df['device'].unique():
            sub = df[df['device'] == device]
            show_legend = device not in seen_devices
            seen_devices.add(device)
            fig.add_trace(
                go.Bar(
                    y=sub['device'],
                    x=sub['duration_us'],
                    base=sub['start_us'],
                    orientation='h',
                    name=device,
                    legendgroup=device,
                    showlegend=show_legend,
                    marker_color=color_map.get(device, '#888'),
                    customdata=sub[['name', 'duration_us', 'kernels']].values,
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'Duration: %{customdata[1]:.4f} µs<br>'
                        'Kernels:<br>%{customdata[2]}<extra></extra>'
                    ),
                ),
                row=row, col=1,
            )

        # Header y in paper coords: centre of the reserved header band
        # Bold title
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.5, y=header_y + 0.012,
            text=f'<b>Run forward call {n}</b>',
            showarrow=False, xanchor='center',
            font=dict(size=15, color='#111'),
        )

        # Token budget boxes
        box_h = 0.018
        box_w = 0.022
        gap   = 0.005
        x0    = 0.01
        by    = header_y - 0.022
        
        for i in range(token_budget):
            bx0 = x0 + i * (box_w + gap)
            bx1 = bx0 + box_w
            color = TOKEN_PREFILL_COLOR if i < m else (TOKEN_DECODE_COLOR if i < m + k else TOKEN_EMPTY_COLOR)
            fig.add_shape(type='rect',
                xref='paper', yref='paper',
                x0=bx0, x1=bx1, y0=by, y1=by + box_h,
                fillcolor=color, line=dict(color='#aaa', width=1),
            )
        
        if m > 0:
            fig.add_annotation(
                xref='paper', yref='paper',
                x=x0 + (m / 2) * (box_w + gap) - gap / 2,
                y=by + box_h / 2,
                text='PREFILL', showarrow=False,
                font=dict(size=8, color='#333'),
            )
        for i in range(m, m + k):
            fig.add_annotation(
                xref='paper', yref='paper',
                x=x0 + i * (box_w + gap) + box_w / 2,
                y=by + box_h / 2,
                text='D', showarrow=False,
                font=dict(size=8, color='#333'),
            )

        tx = x0 + token_budget * (box_w + gap) + 0.01
        fig.add_annotation(
            xref='paper', yref='paper',
            x=tx, y=by + box_h / 2,
            text=f'pref={m}, dec={k}',
            showarrow=False, xanchor='left',
            font=dict(size=10, color='#555'),
        )

    fig.update_layout(
        barmode='overlay',
        height=ROW_HEIGHT_PX * n_runs,
        hovermode='closest',
        legend_title='Device',
        margin=dict(t=30, b=40),
    )
    fig.show()

import json
import re
import csv
from pathlib import Path
from collections import defaultdict

def gen_summary(path, event_name):
    trace_dir = Path(path)
    durations = defaultdict(list)

    for path in trace_dir.glob("run*_*.json"):
        m = re.match(r"run\d+_(.+)\.json", path.name)
        if not m:
            continue
        workload = m.group(1)

        with open(path) as f:
            trace = json.load(f)

        events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
        for ev in events:
            if ev.get("name") == event_name and "dur" in ev:
                if ev["dur"] != 0:
                    durations[workload].append(ev["dur"])

    out = trace_dir / f"avg_latencies_{event_name}_3.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "avg_dur", "count"])
        for workload, durs in sorted(durations.items()):
            w.writerow([workload, sum(durs) / len(durs), len(durs)])
