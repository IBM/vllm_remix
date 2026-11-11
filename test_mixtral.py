import os
#os.environ["VLLM_USE_FUSED_LAYERNORM"] = "0"
#os.environ["VLLM_NO_FUSED_OPS"] = "1"
#os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
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
    p.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-v0.1")
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
    CHUNKED_PREFIL = False
    GPU_UTILIZATION=0.9
    ASYNC_SCHEDULING = True
    model_id = args.model 
    number_gpus = 1
    #if fixed map
    with open('yaml/mixtral_partition.yaml') as f:
        allocation_map = yaml.safe_load(f)
    #run DP and explore partition
    with open("yaml/mixtral_device_exploration.yaml") as f:
         device_map = yaml.safe_load(f)
    llm = LLM(
        model=model_id,
        #tensor_parallel_size=number_gpus,
        load_format="dummy",
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
            mode="VLLM_COMPILE",
            cudagraph_capture_sizes = [1,4,8,16],
            device_exploration_map =  device_map,
            default_device = "h100",
            available_devices =["h100", "3dcim", "dram-pim"],
            custom_ops = ["none"],
        ),
        profiler_config = ProfilerConfig(profiler="remix_aggregator",
                                transfer_latencies_path = 'yaml/transfer_latencies.yaml',
                                chrome_trace_path = "mixtral/run.json",
                                cuda_traces_path="mixtral/run/json"),
    )
    vocab_size = llm.llm_engine.model_config.hf_config.vocab_size
    torch.manual_seed(0)
    warmup_prompt = TokensPrompt(prompt_token_ids=torch.randint(0, vocab_size, (isl+1,)).tolist())
    # warm up
    # bypassing the population overhead of the SSM (?)
    llm.generate(prompts=[warmup_prompt], sampling_params=SamplingParams(temperature=0.0, min_tokens=1, max_tokens=1, ignore_eos=True))
    prompts = [TokensPrompt(prompt_token_ids=torch.randint(0, vocab_size, (isl,)).tolist()) for _ in range(n_queries)]
    assert len(prompts[0]["prompt_token_ids"]) == isl
    sampling = SamplingParams(temperature=0.0, max_tokens=osl, ignore_eos=True)
    llm.start_profile()
    output = llm.generate(prompts=prompts, sampling_params=sampling)
    llm.stop_profile()


if __name__ == "__main__":
    main()