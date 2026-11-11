import os
os.environ["VLLM_USE_FUSED_LAYERNORM"] = "0"
os.environ["VLLM_NO_FUSED_OPS"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
import torch

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig,ProfilerConfig,AttentionConfig
from vllm.inputs import TokensPrompt
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum)
import logging
import yaml

GPU_UTILIZATION = 0.95
CHUNKED_PREFIL = False
ASYNC_SCHEDULING = False

import argparse
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="ibm-ai-platform/Bamba-9B-v1")
    p.add_argument("--isl", type=int, default=128)
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--bs", type=int, default=1)
    args = p.parse_args()
    isl = args.isl
    osl = args.osl 
    n_queries = args.bs
    TOKEN_BUDGET = isl+osl
    MODEL_CONTEXT = isl+osl # min OSL+ISL
    assert TOKEN_BUDGET >= MODEL_CONTEXT
    model_id = args.model 
    number_gpus = 1
    with open('yaml/bamba/remix_partition_afd.yaml') as f:
        allocation_map = yaml.safe_load(f)
    with open("yaml/bamba_device_exploration.yaml") as f:
         device_map = yaml.safe_load(f)
    llm = LLM(
        model=model_id,
        #tensor_parallel_size=number_gpus,
        load_format="dummy",
        hf_overrides={"num_hidden_layers": 10},
        pipeline_parallel_size = number_gpus,
        gpu_memory_utilization=GPU_UTILIZATION, 
        max_model_len= MODEL_CONTEXT,            # Limit context to reduce KV cache
        max_num_batched_tokens = TOKEN_BUDGET, # token budget of the SCHEDULER
        max_num_seqs = n_queries, # ? not sure
        async_scheduling = ASYNC_SCHEDULING,
        enable_chunked_prefill =CHUNKED_PREFIL,
        enable_prefix_caching = False,
        disable_log_stats = False,
        compilation_config = CompilationConfig(
            cudagraph_mode= "PIECEWISE",
            backend="remix",
            cache_dir = '/u/mvc/vllm_remix/vllm_cache',
            mode="VLLM_COMPILE",
            cudagraph_capture_sizes = [1, 2, 16],
            device_exploration_map =  device_map,
            #splitting_dict = allocation_map,
            default_device = "h100",
            available_devices =["h100", "dram-pim", "3dcim"],
            custom_ops = ["none"],
            #compile_sizes= [1,20],
        ),
        #profiler_config = ProfilerConfig(profiler="torch", torch_profiler_dir="/u/mvc/remix_vllm/profiler"),
        profiler_config = ProfilerConfig(profiler="remix_aggregator",
                                transfer_latencies_path = 'yaml/transfer_latencies.yaml',
                                chrome_trace_path = f'bamba/run.json',
                                cuda_traces_path=f'bamba/run.json'),
        #enforce_eager=False,
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN),
    )
    vocab_size = llm.llm_engine.model_config.hf_config.vocab_size
    prompts = [TokensPrompt(prompt_token_ids=torch.randint(0, vocab_size, (isl,)).tolist()) for _ in range(n_queries)]
    assert len(prompts[0]["prompt_token_ids"]) == isl
    sampling = SamplingParams(temperature=0.0, max_tokens=osl, ignore_eos=True)
    # All profiling will only occur if calling llm.start_profile like the torch profiler intended
    llm.start_profile()
    output = llm.generate(prompts=prompts, sampling_params=sampling)
    llm.stop_profile()

if __name__ == "__main__":
    main()