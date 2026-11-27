# inference-optimisation

> English version: [README.md](README.md)

Một skill cho agent chuyên **leo dốc tối ưu tốc độ của một máy chủ LLM chạy local**, trên mọi phần cứng và mọi kiểu lượng tử hoá. Skill bao quát:

- decode cho một người dùng và decode đồng thời nhiều người dùng
- speculative decoding (giải mã suy đoán)
- prefill
- kích thước KV cache
- thời gian khởi động và thời gian tới token đầu tiên (TTFT)

Skill hoạt động dưới một giới hạn bộ nhớ, và mọi kết quả đều phải vượt qua bài kiểm tra độ ổn định ở toàn bộ ngữ cảnh.

Skill chắt lọc cách chúng tôi thực sự đạt được những mức tăng tốc đó. Nó không phải một danh sách các cờ lệnh. Quy trình làm việc:

1. Tính trần hiệu năng lý thuyết.
2. Mỗi lần chỉ thay đổi một thứ.
3. Đo đối chứng xen kẽ với các bản kiểm soát.
4. Ghi lại mọi kết quả, kể cả những lần thất bại.
5. Kiểm tra bằng bài kiểm tra truy hồi ở ngữ cảnh dài.
6. Thẩm định một bản build sạch, đã ghim phiên bản, trước khi công bố.

Skill hoạt động với Claude Code và mọi agent có khả năng nạp skill kiểu `SKILL.md`. Mọi model cũng có thể làm theo từ [`PROMPT.md`](PROMPT.md).

## Những kết quả đã đạt được

Kết quả từ những đợt leo dốc tối ưu mà skill này được chắt lọc ra. Mọi con số đều được đo trên phần cứng ghi rõ; xem các nguồn liên kết để biết phương pháp và dữ liệu thô.

| Phần cứng | Model / lượng tử hoá / engine | Trước → sau |
|---|---|---|
| **Intel Arc B390 iGPU** (laptop, 64 GB LPDDR5X hợp nhất, ~115 GB/s) | Ornith-1.5-35B-A3B, EXL3 4 bpw, SGLang XPU + MTP | Decode, 1 người dùng: 43.6 → **53 tok/s** (trung vị)<br>Decode, 4 người dùng: 84.7 → **103 tok/s**<br>Prefill 32K: 1,037 → **1,660 tok/s**<br>Prefill 256K: 297 → **628 tok/s** (2.1×)<br>KV cache: 264K tokens → **4 × 256K**<br>Khởi động: 113 → **43 s**<br>Cổng kiểm tra Registry Lab: decode 28.6 → **41.0**, prefill 255 → **667** |
| **Intel Arc Pro B70** | Qwen3.8-27B, EXL3 4 bpw, vLLM + kernel XPU tuỳ chỉnh | Decode: **91.2 tok/s** với 1 người dùng, **365** với 16 (văn xuôi, bật thinking)<br>Cùng card với llama.cpp Q4_K_M: 25.0<br>Prefill 4K/32K/128K/254K: 1,654/1,434/928/650 → **2,421/2,259/1,415/911**<br>Phát lại phiên agent: 193 → **147 s** |
| **Intel Arc Pro B70** | Cùng model, SGLang + cùng các kernel đó | Decode, 1 người dùng: **82.6 tok/s**, so với vLLM 75.7 trên cùng card và cùng ngày |
| **RTX 3090** | Qwen3.8-27B EXL3 3 bpw + MTP | **96.2 tok/s** văn xuôi / **141.1** code, 1 người dùng |
| **RTX 3090** | Qwen3.6-35B-A3B EXL3 3 bpw + MTP | **251.7 / 357.9 tok/s**, 1 người dùng |
| **RTX 3090** | Flash-Next EXL3, ngữ cảnh 256K | MTP: 45.5 → **65.6 tok/s** |
| **4× H200 NVL** | DeepSeek-V4.1-Flash, vLLM + speculative decoding | Sập khi chịu tải → **229.1 tok/s** decode, **17,546 tok/s** prefill; cả sáu cổng kiểm tra đều đạt |

Nguồn:

- [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry):
  - [Công thức Ornith B390](https://github.com/sybil-solutions/local-ai-registry/tree/main/data/sources/ornith-b390-exl3), với nhật ký leo dốc đầy đủ trong [PR #453](https://github.com/sybil-solutions/local-ai-registry/pull/453)
  - các công thức RTX 3090 và H200
- [trellis-serve](https://github.com/sybil-solutions/trellis-serve):
  - `xpu/README.md`, `xpu/docs/PROGRESS.md` và `xpu/docs/SGLANG.md` cho B70
  - `cuda/README.md` cho RTX 3090
- [qwen-serve](https://github.com/sybil-solutions/qwen-serve)

### Cách đọc kết quả B390

Đợt chạy đó là nơi skill này ra đời. Phần tính trần hiệu năng của nó cho thấy:

- **Decode:** 100 tok/s cho một người dùng chỉ đạt được khi có hai bản nháp suy đoán hữu ích. Đầu nháp (draft head) của model này chấp nhận ~14% ở độ sâu 2, nên mục tiêu thực tế là ~88–97 tok/s. Chúng tôi đạt ~55% của mức đó; phần còn lại nằm ở hiệu quả của kernel GEMV.
- **Prefill ở 256K:** chúng tôi đạt 628 tok/s so với trần ~670–684 tok/s.

Skill bắt agent tính việc này ngay từ đầu, để nó có thể cho bạn biết sớm khi một mục tiêu nằm ngoài giới hạn vật lý.

## Cách hoạt động

```
                ┌────────────────────────────────────────────────────────────────┐
  measure HW ─▶ │ THEORY.md: bytes/token ÷ bandwidth, FLOPs/token ÷ 50% peak, KV │
                └────────────────────────────────────────────────────────────────┘
                                         │
    baseline ─▶ profile ─▶ pick lever (ceiling gap × likelihood) ─▶ configs/NEW.json
                    ▲                                                     │
                    │                                    queue.sh ctl NEW ctl (serialized, AC-only,
                    │                                    memory watchdog, ≥6 samples, medians)
                    │                                                     │
   PLAN.md + ledger.jsonl ◀── accept only if it beats BOTH neighbouring controls ──┘
          │                    and regresses nothing by >3%
          ▼
  long-context needle gate ─▶ concurrent full-KV soak ─▶ plateau? ─▶ pinned clean build
                                                                      ─▶ acceptance gates ─▶ publish
```

Skill gồm ba phần:

**[`SKILL.md`](skills/inference-optimisation/SKILL.md): quy trình.**
- Kiểm kê, trần hiệu năng, luật nền, vòng lặp và thứ tự các đòn bẩy.
- Khi nào dừng, cách thẩm định và công bố, và cách báo cáo.

**[`references/`](skills/inference-optimisation/references): chỉ nạp khi cần.**
- [`ceilings.md`](skills/inference-optimisation/references/ceilings.md): cách tính cực đại của decode, prefill, TTFT, khởi động và KV, kèm một ví dụ đã làm sẵn.
- [`methodology.md`](skills/inference-optimisation/references/methodology.md): các cạm bẫy đo lường. Token đếm theo chunk, prompt sai kích thước, bộ đếm chấp nhận, độ trôi, cách hạch toán bộ nhớ hợp nhất.
- [`levers.md`](skills/inference-optimisation/references/levers.md): mọi đòn bẩy kèm bằng chứng đã đo. Được gắn nhãn dùng chung hay riêng cho từng stack.
- [`gotchas.md`](skills/inference-optimisation/references/gotchas.md): các cạm bẫy đã biết trong vLLM, SGLang, llama.cpp và ExLlama, và trên phần cứng Intel, NVIDIA, Apple và AMD.
- [`publishing.md`](skills/inference-optimisation/references/publishing.md): ghim phiên bản, build sạch, sáu cổng kiểm tra chấp nhận, chứng minh bản build khớp với bản tốt nhất, thẻ kết quả.

**[`scripts/`](skills/inference-optimisation/scripts): một harness nhỏ, không phụ thuộc engine.** Nó là Python thư viện chuẩn cộng với bash.
- `bench.py` đo benchmark mọi máy chủ tương thích OpenAI. Nó đo:
  - khởi động và TTFT với prompt ngắn;
  - decode c1 (trung vị của N lần) và decode tổng hợp cN;
  - prefill nguội ở nhiều độ dài;
  - truy hồi "kim trong đống rơm" ở ngữ cảnh dài;
  - bộ nhớ đỉnh, kèm cơ chế giám sát giới hạn.

  Nó gắn cờ đầu ra bị lặp vòng, các request thất bại và các lần chạy bằng pin. Số token lấy từ thống kê usage của máy chủ, không bao giờ từ các chunk truyền về, và mọi thứ đều được ghi vào `ledger.jsonl`.
- `ceiling.py` tính các trần hiệu năng lý thuyết từ một file JSON chứa băng thông và FLOPs đã đo cộng với kích thước các thành phần.
- `queue.sh` là một hàng đợi chạy nối tiếp, tách rời được, để xen kẽ các lần chạy kiểm soát/ứng viên/kiểm soát.
- `mem.sh` báo cáo dung lượng thực của một container. Con số này gồm cả phần driver chiếm dụng trên các GPU bộ nhớ hợp nhất, điều mà cgroups không nhìn thấy.

## Cài đặt

Claude Code (mức người dùng):

```bash
git clone https://github.com/sybil-solutions/inference-optimisation-skill
mkdir -p ~/.claude/skills
cp -r inference-optimisation-skill/skills/inference-optimisation ~/.claude/skills/
```

Sau đó hãy yêu cầu, ví dụ:

- "leo dốc tối ưu decode và prefill cho model ở :8000 dưới giới hạn 40 GiB"
- "tốc độ tok/s tối đa theo lý thuyết của model này trên máy này là bao nhiêu?"

Các agent khác: trỏ chúng tới `skills/inference-optimisation/SKILL.md`, hoặc dán [`PROMPT.md`](PROMPT.md) làm prompt nhiệm vụ.

## Bắt đầu nhanh không cần agent

```bash
S=skills/inference-optimisation
python3 $S/scripts/ceiling.py $S/examples/ceiling-ornith-b390.json        # ceilings
mkdir -p configs && cp $S/examples/config-existing-server.json configs/base.json   # edit base_url/model
python3 $S/scripts/bench.py run configs/base.json --long 131072            # one run
bash $S/scripts/queue.sh base cand base                                     # interleaved A/B
python3 $S/scripts/bench.py show                                            # ledger table
```

## Những nguyên tắc có trọng lượng nhất

- **Trần hiệu năng trước tiên.** Nếu không biết mức tối đa, bạn không thể nói 53 tok/s là tốt hay tệ, hay "100 tok/s" có khả thi hay không.
- **Nhiễu là ±5–10%.** Hầu hết các "thắng lợi" trong một lần chạy đơn lẻ đều là độ trôi. Hãy xen kẽ các bản kiểm soát và dùng trung vị.
- **Đo thứ quan trọng:**
  - token thật, không phải chunk truyền về;
  - độ dài prompt thật, từ tokenizer;
  - bộ nhớ thật, gồm cả phần GPU chiếm dụng trên bộ nhớ hợp nhất.
- **Speculative decoding là chuyện chi phí mỗi bước, không phải tỉ lệ chấp nhận.**
  - Cắt bớt từ vựng của đầu nháp.
  - Quét độ dài bản nháp theo từng mức đồng thời và ngữ cảnh.
  - Giữ việc lấy mẫu (sampling) trên GPU.
- **Chứng minh độ ổn định ở toàn bộ ngữ cảnh** bằng các "kim" khớp chính xác và một bài ngâm tải toàn bộ KV đồng thời, trước khi gọi bất cứ thứ gì là "tốt nhất".
- **Viết lại các thất bại.** Phần lớn giá trị trong `PLAN.md` là danh sách những thứ đã *không* hiệu quả và lý do.

## Đóng góp

Hoan nghênh các PR bổ sung đòn bẩy, cạm bẫy hoặc kết quả mới. Mỗi PR cần ghi rõ phần cứng, model, lượng tử hoá, phiên bản engine, số liệu trước → sau, cách đo, và liên kết tới dữ liệu thô.

## Giấy phép

MIT
