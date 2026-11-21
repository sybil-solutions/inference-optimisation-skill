# Multi-GPU and multi-node serving

Read this only when one model is split over several GPUs or hosts. Everything else in the skill still applies; this adds the links between ranks.

## Before measuring

- **A degraded link raises no error.** Bandwidth-bound work just runs slower, and any A/B that depends on it is wrong. On every node, every session, check each fabric port's negotiated speed, MTU, carrier and address, that its RDMA port is `ACTIVE` (`rdma link`), and that the kernel log shows no NIC re-probe since boot.
- **Check what the collective library chose.** With `NCCL_DEBUG=INFO`, look for `NET/IB`, not `NET/Socket`, and for every device you listed in `NCCL_IB_HCA`.
- **Once a session, run `ib_write_bw` per link against a floor.** A throttled NIC can keep its link up at full speed; only this test or the kernel log shows it.

## Communication time

- **Measure each collective the engine runs** (all-gather, all-reduce, ...) at the message sizes it sends, with nccl-tests or rccl-tests. Sum them per forward: latency dominates at decode sizes, bandwidth at prompt sizes. It is an estimate, not a bound.
- **Bus bandwidth units:** algbw × (n − 1)/n for an all-gather, × 2(n − 1)/n for an all-reduce; quoting algbw as bus bandwidth is a 4/3 error at 4 ranks. nccl-tests prints an all-gather's size as the total over ranks, 4× the per-rank size at 4 ranks.
- **Profile collectives as transfer against wait.** Transfer is a collective's shortest duration across ranks, a lower bound on the network's cost; wait is each rank's excess over it. Wait ≫ transfer points to load imbalance or a serial rank before the network.
  - TP4 on 4× GB10, one profile on a ring with one throttled NIC, eager, whole experts per rank: all-gathers took 13.2–15.4 ms of a 68 ms decode step, only ~1.6 ms of it transfer.
- **Wait in the MoE gathers can be the expert placement.** With whole experts on each of 4 ranks, top-8 routing puts ~3.5 picks on the busiest rank against a mean of 2 even when routing is uniform. In the profile above, 10–11.5 ms of the wait was in the MoE gathers. A balanced layout was not measured end to end.

## Measuring

- **Compare only within one session,** with every node preflighted and each arm loaded twice: `queue.sh ctl-a cand-a cand-b ctl-b`, with two copies of each config under different names. The fabric may have changed since an earlier session.
- **Splitting changes the bits.** Partial sums can differ from one GPU in the last bits at any rank count, so run the greedy equivalence check against the split build. Whether runs reproduce depends on the order of the cross-rank sum: at 2 ranks it is one add (`a + b == b + a`); from 3 ranks, fix the rank order.

## Levers

- **[G] Fix a degraded link first: a ring collective runs at its slowest link.** 4× GB10 / NCCL over RoCE: a NIC re-probed after a cable move sent ~10 Gb/s per PCIe half against ~98 for the others with every rail sending. Prompt-size all-gathers ran at 3.1 GB/s bus bandwidth on that ring and 22.1 GB/s on a switch with every host rebooted; no healthy ring was measured. Small decode collectives barely changed.
- **[S: NCCL over RoCE] List every RDMA device a port exposes in `NCCL_IB_HCA`.** A GB10's 200G port is two PCIe functions. Qwen3.8-Flash-Next TP2 on two GB10s on one cable, NCCL 2.32.3: `ib_write_bw` 111.9 Gb/s on one device, 98.0 + 98.0 on both; cold prompts 26.5–28.1% faster (mean of 2 starts), same hashes.
- **[S: TensorFold] If you can change the engine, reduce before you gather.** When a local reduction follows the cross-rank sum, reduce-scatter, reduce your own rows, then all-gather the smaller result. Qwen3.8-Flash-Next TP2: rank 0 sent 5.81 → 3.43 MiB per prompt token on two GB10s; cold prefill +26.5–26.8% there (RoCE, one start each) and +30.9–33.9% on two RTX PRO 6000s in one host (PCIe, one start against two), not alternated, same bits.

## Gotchas

- **NCCL hangs at its first collective, with init done and QPs connected:** suspect a firewall on the bootstrap or socket interface (`NCCL_SOCKET_IFNAME`).
- **NVIDIA GB10 (DGX Spark) with ConnectX-7:**
  - A runtime re-probe (a moved cable) caps transmit at ~10–13 Gb/s per PCIe half until reboot, with the link still up at 200G. The kernel logs `insufficient power on the PCIe slot` or `PCIe slot power capability was not advertised`; the boot probe logs it too, so count only later lines. Reboot after any cable change.
  - A re-probe can bring ports back down, with MTU 1500 and no address.
