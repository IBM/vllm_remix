# vLLM-REMIX

A fork of [vLLM](https://github.com/vllm-project/vllm) that adds a heterogeneous-device partitioning framework called **REMIX** on top of vLLM's `torch.compile`-based piecewise compilation backend. REMIX lets you split a single LLM graph across multiple device types — GPU, attention accelerators, PIM, custom NPUs — and route each subgraph to the backend that runs it most efficiently.


## Overview

Standard vLLM compiles the full model graph with a single backend (Inductor). REMIX intercepts the compiled `torch.fx` graph, applies module-level partition rules defined in a YAML file, splits the graph into subgraphs, and dispatches each subgraph to a pluggable `BaseBackend`. Device transfers between subgraphs are handled transparently.

Key additions over upstream vLLM:

| Feature | Description |
|---|---|
| YAML partition rules | Assign any named module (or glob pattern) to a named device |
| Multi-device dispatch | Each subgraph runs on its own backend with automatic host/device tensor transfers |
| Pluggable backends | `CUDAInductorBackend`, `AttAccBackend`, `TRACEBackend`, `PassthroughBackend` |
| Performance databases | CSV-based lookup tables for GEMM, attention, and MoE latencies on specific hardware |
| Timing collector | Per-subgraph latency instrumentation that writes JSON traces |
| Simulation mode | Estimate end-to-end latency from performance databases without running on real hardware |

---

## REMIX Installation

### Prerequisites

- Linux with an NVIDIA GPU and a working driver
- Python 3.12
- `micromamba` or `conda`, and `uv`
- Graphviz binaries (`dot`), required by the compilation backend

### Steps

1. Create and activate an environment:

```
micromamba create -n remix python=3.12 -y
micromamba activate remix
```

2. Install `uv` and Graphviz:

```
pip install uv
micromamba install -c conda-forge graphviz -y
```

3. Clone the repository:

```
git clone <REPO_URL>
cd vLLM-REMIX
```

4. Pin the upstream vLLM commit whose precompiled binaries match this code:

```
export VLLM_PRECOMPILED_WHEEL_COMMIT=9ef3b718d98ada5c02ebe67c2a545d3005a6b10d
export VLLM_USE_PRECOMPILED=1
```

5. Install in editable mode:

```
uv pip install -v --editable . --torch-backend=auto
```

6. Verify the install:

```
python -c "import vllm._C; print('ok')"
dot -V
```

## Citation / Upstream


> **TODO**: add a citation for the REMIX work once published.
