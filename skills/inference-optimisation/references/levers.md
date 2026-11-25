# Levers, by area, with measured evidence

**[G]** general; **[S: stack]** seen on one stack. Numbers come from real hill-climbs (see the README): priors for your forecast, not promises.

## Startup and TTFT

- **[G] Persist every JIT, compile and autotune cache** (SYCL/NEO/Triton/Inductor, DeepGEMM, `~/.cache`): B390 / SGLang 113 → 43 s; 4× RTX PRO 6000 / vLLM ~22 → 7 min.
- **[S: vLLM] mmap loading ran ~60 MB/s on some filesystems** (26 min for 27 GB); `--safetensors-load-strategy eager` took seconds.
- **[G] One compile worker** (`TORCHINDUCTOR_COMPILE_THREADS=1`) on memory-tight machines; workers held ~6 GiB transient.
- **[G] Short-prompt TTFT is scheduling plus one forward pass:** check tokenizer, template and sampler before kernels.

## Decode

- **[G] Decode graphs first:** Arc Pro B70 14.4 → 28.4 tok/s; SGLang XPU 16 → 32; vLLM INT4 11 → 27.
- **[G] Speculative decoding, measured per step.**
  - **Prune the draft LM head's vocabulary**; the target verifies against the full vocabulary, so output is unchanged. B70 / vLLM c1 37.9 → 48.2 (+27%) with 512 of 1,940 blocks (98.6% coverage); B390 / SGLang +14% with 256.
  - **Best draft length depends on concurrency and context:** B70 k=4 at c1, 3 at c4, 2 at c16 (+11%, more streams fit); B390 at 128K 1 draft 29.6 vs 15.5 for 3, though 3 won at short context.
  - **Some MTP heads are weak beyond depth 1** (Ornith-1.5: ~14% at depth 2). Measure acceptance per depth.
  - **Speculation can lose** on bandwidth-poor devices or hard prompts: Apple M4 6.35 → 3.25 with 3 MTP drafts; llama.cpp MTP 36.7 vs 45.4; a CPU-offloaded MoE 0.35–0.5×.
  - **Verify with decode-style attention.** k-query verifies on the prefill kernel grew step time 3.5 ms per 1K context (47 → 104 ms at 16K); as decode queries SGLang B70 c1 went 75.7 (vLLM) → 82.6.
  - **Keep sampling on the GPU:** a CPU sampler cost ~3 ms a step (+10% c1 on B390 when moved).
- **[G] Fewer bytes per weight when bandwidth-bound,** in a natively decoded format: on an RTX 3090 AWQ-INT4 beat FP8.
- **[G] Thread counts and split-K tuned for one card are wrong for another:** B70 4096 → 1024 threads took code c1 48.5 → 56.9.
- **[G] Under graphs, fusing small kernels rarely helps:** three correct fusions on B390 gave 0%. In a batched step, time followed bytes (R² ~0.87), not node count.
- **[G] KV dtype is a trade.** FP8 KV halved one flash-attention kernel's throughput (39 vs 69–74 TFLOPS) until blocks were dequantized to fp16 with an LSE merge (128K prefill 760 → 1,059). With KV in 16 of 64 layers, q8_0 KV went 45.4 → 27.0 on llama.cpp.

## Batched serving and the host path

- **[G] Take host work off the step's critical path.** A per-turn state publish deep-copied on the command thread was 8–11% of a stage's wall; moving it gave +11–12% decode at 16 and 32 streams.
- **[G] Where prompts run is a throughput/latency trade.** Long cold prompts inside batched steps starved decode (−20% to −60%); a third of requests on a side path drained the batch (−8% to −18%) but cut first-token time ~7×.
- **[G] Smaller or more numerous pipeline steps lose when each step has a fixed cost:** four groups instead of two −11% to −28%.
- **[G] Tiering cost follows the share of steps touched, not of picks.** The coldest experts (2.8% of picks) on a remote card over a 56 Gb/s link cost −44%: 98% of layer steps carried a remote pick and waited for the hop.
- **[G] CPU expert tiers are read-bound.** Memory interleaved over all NUMA nodes +56–118% at one stream; 32 threads beat 64–128; huge pages added nothing.

## Concurrency and memory

- **[G] Hybrid SSM/GDN models: state slots and page size cap concurrency, not KV.** SGLang ran 1 request until `--max-mamba-cache-size` rose; vLLM's exact 1,600-token block (not 2,048) took c16 from 10 to 14 streams (+17%).
- **[G] Size KV to N × context; prove it with a concurrent full-length soak:** B390 264K → 1.05M tokens (4 × 256K), 4/4 needles at 36.8 GiB.
- **[G] Enable prefix caching explicitly** and confirm `cached_tokens` > 0: turn-2 TTFT 36 → 0.45 s at 66K, 29 → 0.95 s at 32K.
- **[G] Memory outside the framework allocator causes OOMs** (scratchpads, a primitive per shape, fragmentation): route scratch through it, bucket shapes, `expandable_segments:True`.
- **[G] Leave headroom:** over-allocating on Arc livelocked the box; keep ≥1.5 GB free.

## Prefill

- **[G] Chunk size trades prefill throughput against others' decode stalls:** B70 8,192 → 2,048 cut the worst stall 7.1 → 1.95 s for −4% decode; B390 1,024 → 4,096 with INT8 GEMMs +50% at 32K.
- **[G] Swap in a faster attention library before writing one:** oneDNN fused SDPA +7–18% over FA2 at 4K–254K; a hand-written one reached 48 vs 61 TFLOPS.
- **[G] Int8/fp8 prefill GEMMs (labelled lossy):** with Hadamard rotation per-row scales suffice; linears 1.89×, NLL +0.17%, top-1 97.2%, decode bit-exact.
- **[G] Grouped MoE GEMMs: avoid padding cliffs, reuse decoded weights.** Blocks for M = 17–48 −5%; a 64-row M-block +7% end-to-end, bit-exact; an NT=2 tile fix +22% at c16 on B70.
- **[G] Profile by stage first:** in one 32K prefill GEMMs were 40% and already at the int8 ceiling.

## Kernel work (after config levers plateau)

- Test bit-exact or tight-tolerance against the old kernel at every row count M you serve; a "−4%" kernel once gave wrong outputs a two-path test missed. Lossless ladder: weights bit-identical → each layer vs fp64 → graph vs eager → model KL.
- Read the ISA: on B70 stride-3 register regions became per-element moves; making them planar took the LM head 4.05 → 1.72 ms.
- Check memory-model fences on new GPUs: split-K partial stores on Xe3 needed L1-uncached stores plus a cache-clean fence, or results were silently wrong at some M.
- Rejected on Intel Xe: fused single-kernel linear (serialized tail), software pipelining (spills, M=16 39 → 86 ms), lookup-table codebooks (4× slower than ALU decode), LSC prefetch (noise), async scheduling (0%).
