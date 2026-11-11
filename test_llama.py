import os
os.environ["VLLM_USE_FUSED_LAYERNORM"] = "0"
os.environ["VLLM_NO_FUSED_OPS"] = "1"
import torch

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig,ProfilerConfig,AttentionConfig
from vllm.inputs import TokensPrompt
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum)
import logging
import yaml

import argparse
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--isl", type=int, default=128)
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--bs", type=int, default=1)
    args = p.parse_args()
    isl = args.isl
    osl = args.osl 
    n_queries = args.bs
    GPU_UTILIZATION = 0.9
    TOKEN_BUDGET = isl+osl
    MODEL_CONTEXT = isl+osl # min OSL+ISL
    assert TOKEN_BUDGET >= MODEL_CONTEXT
    CHUNKED_PREFIL = True
    ASYNC_SCHEDULING = False
    model_id = args.model
    number_gpus = 1
    with open('yaml/llama/remix_partition_3way.yaml') as f:
        allocation_map = yaml.safe_load(f)
    with open("yaml/llama_device_exploration.yaml") as f:
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
        disable_log_stats = False,
        compilation_config = CompilationConfig(
            cudagraph_mode= "PIECEWISE",
            backend="remix",
            cache_dir = './vllm_cache',
            mode="VLLM_COMPILE",
            cudagraph_capture_sizes = [1,2,4,8],
            #splitting_dict = allocation_map,
            device_exploration_map =  device_map,
            default_device = "h100",
            available_devices =["h100", "dram-pim", "3dcim"],
            custom_ops = ["none"]
        ),
        #profiler_config = ProfilerConfig(profiler="torch", torch_profiler_dir="/u/mvc/remix_vllm/profiler"),
        profiler_config = ProfilerConfig(profiler="remix_aggregator",
                                transfer_latencies_path = 'yaml/transfer_latencies.yaml',
                                chrome_trace_path = f'3way_llama_bs{n_queries}_isl{isl}_osl{osl}/run.json',
                                cuda_traces_path=f'llama_dump/run.json'),
        #enforce_eager=False,
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN),
    )
    llm.start_profile()
    vocab_size = llm.llm_engine.model_config.hf_config.vocab_size
    prompts = [TokensPrompt(prompt_token_ids=torch.randint(0, vocab_size, (isl,)).tolist()) for _ in range(n_queries)]
    assert len(prompts[0]["prompt_token_ids"]) == isl
    output = llm.generate(prompts=prompts, sampling_params=SamplingParams(max_tokens=osl,  min_tokens = osl))

if __name__ == "__main__":
    main()