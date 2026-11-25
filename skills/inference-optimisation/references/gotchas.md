# Engine and hardware gotchas

Found the hard way; re-check on newer releases.

## General

- **A second engine or stray benchmark on the GPU** silently halves results or crashes both. Lock it (`scripts/queue.sh`).
- **Correlate "device lost" crashes with the kernel log before blaming code.** Arc Pro B70 crashes were PCIe link drops (AER replay timeouts, link down, 8 times in 3 days) that spread to devices on the same root complex.
- **Forced container stops during a long prefill** caused an Intel xe GPU page fault. Abort, drain, then stop with a timeout.
- **A launch that returns before the server is up** (`docker run -d`, a detached worker) hides a dead server behind a long health timeout; probe liveness too. A worker handshake once hit its 1 s deadline while the worker still loaded; a ready line from the worker now gates the engine start.
- **Huge images can crash vision encoders;** cap the longest edge.
- **Gateway timeouts** (often 600 s) cut off long-context tests. Test the engine directly; retry dropped connections, never timeouts.
- **`pkill -f` can kill its own shell,** and `pgrep -f` can wait on itself. Kill by PID.
- **Agent restarts kill child processes, and `nohup`/`setsid` jobs are invisible to harness and user.** Use harness background tasks or units (`systemd-run --user --unit=NAME`); make jobs resumable.
- **A new binary's first load can JIT for minutes** (~7.5 once): exclude it from startup numbers; size the health deadline for it.

## vLLM

- Prefix caching may be off by default for hybrid (SSM/GDN) models. Enable it and check `cached_tokens`.
- On Intel XPU only `FULL_DECODE_ONLY` graphs worked; piecewise capture (forced by dynamic draft length) failed with `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
- With MTP, streaming sends several tokens per chunk. Use usage stats.
- mmap loading can be very slow; use `--safetensors-load-strategy eager`.
- On unified memory `gpu_memory_utilization` is checked against free host RAM (~0.985 refused).

## SGLang

- Speculative acceptance counters update only when a request finishes.
- Hybrids need `--max-mamba-cache-size` for N requests plus speculative state, or concurrency silently stays at 1.
- The XPU backend (0.5.x) needed patches: greedy-only verify at any temperature, a stale GDN wrapper, CUDA-only mamba-scatter guards, missing fp8 descale in extend, an fp8 Q cast in decode; no prefill graphs.
- `--num-continuous-decode-steps` had no effect with speculation on. Shared-expert fusion is enabled only for some backends.
- Prefill-graph captures of 2,048 tokens used 6.3 GB and crashed; capture up to 256–512.

## llama.cpp / GGML

- Quantized KV (q8_0) can be much slower on models with few KV layers.
- MTP/draft support can fall below the no-draft baseline on hard prompts.
- Check backend work-group sizes: a work-group of 1 in a SYCL copy kernel cost 7.8×.

## ExLlama / TabbyAPI

- On ROCm, HIP graphs leaked until crash; disabled, it was stable at 115.6 tok/s.
- A busy host CPU costs GPU decode on laptops: 107 → 75 tok/s.

## Hardware

- **Intel Xe, PyTorch XPU:** `torch.topk` segfaulted above ~2–8K wide (use staged top-k or Gumbel-max); Xe3 split-K stores need uncached stores plus a fence; `UR_RESULT_ERROR_OUT_OF_RESOURCES` means too-large or piecewise graphs.
- **Unified memory:** GPU memory is invisible to cgroups (`methodology.md`); battery and power profiles move results 8–10%.
- **Apple silicon:** bandwidth-poor chips lose with deep speculation; `iogpu.wired_limit_mb` caps usable memory.
- **NVIDIA Ampere has no FP8 units;** INT4 (AWQ/GPTQ/EXL) wins when bandwidth-bound.
- **Hybrid state dtype:** fp16 GDN/SSM state gave `!!!` garbage on one SGLang XPU build, fine elsewhere. Verify output after changing it.
