#!/usr/bin/env bash
# Production launch as benchmarked (the "mtp2" config). Candidate runs appended their extra flag to the same command;
# "base" is this command without --speculative-config. MODEL is a local abliterated W4A16 build of Qwen/Qwen3.8-Flash-Next.
# vLLM 0.29.1rc1.dev128+gcd10ed6f9 (cu129 nightly) with one local patch, see README.
export VLLM_PLE_CPU_OFFLOAD=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,4
export VLLM_API_KEY=$(cat ~/.vllm-api-key)
exec vllm serve "$MODEL" \
  --served-model-name dgx-current \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --host 127.0.0.1 --port 8000
