# Kế hoạch Thực hiện Lab 22: Căn chỉnh Mô hình Ngôn ngữ bằng DPO/ORPO (Track 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thành trọn vẹn bài Lab 22 về căn chỉnh sở thích con người (DPO/ORPO Alignment) cho mô hình tiếng Việt (Qwen3-4B-Instruct), đạt điểm tối đa 100/100 phần bắt buộc, có đầy đủ bằng chứng thực nghiệm (artifacts, metrics, screenshots) và bài phản tư (`REFLECTION.md`).

**Architecture:** Thực hiện quy trình chuẩn 2 giai đoạn: Huấn luyện có giám sát (SFT-mini) để tạo điểm xuất phát và mô hình tham chiếu (Reference model = `models/sft-merged`), sau đó huấn luyện trực tiếp theo sở thích (DPO) với LoRA mới gắn trên mô hình tham chiếu (precomputed ref log-probs). Đánh giá đối đầu khách quan song song trên tập held-out không trùng câu hỏi bằng hội đồng 2 Reward Models cục bộ.

**Tech Stack:** PyTorch, Unsloth, Hugging Face Transformers, TRL (1.13+), PEFT, Datasets, Matplotlib, Pandas, Qwen3-4B-Instruct-bnb-4bit, Skywork-Reward-V2 (Qwen3-4B & Llama-3.2-3B).

**Spec:** [`README.md`](file:///d:/Document/Ai_Thuc_Chien/8-10-2026/K4-Day22-DaoDuyHieu-2A202602651/README.md), [`rubric.md`](file:///d:/Document/Ai_Thuc_Chien/8-10-2026/K4-Day22-DaoDuyHieu-2A202602651/rubric.md), [`HARDWARE-GUIDE.md`](file:///d:/Document/Ai_Thuc_Chien/8-10-2026/K4-Day22-DaoDuyHieu-2A202602651/HARDWARE-GUIDE.md), [`docs/reference.md`](file:///d:/Document/Ai_Thuc_Chien/8-10-2026/K4-Day22-DaoDuyHieu-2A202602651/docs/reference.md).

---

## Global Constraints

- **Mô hình tham chiếu (Reference Model):** Bắt buộc phải là `models/sft-merged` (mô hình sau khi đã gộp LoRA SFT), KHÔNG được lấy mô hình gốc (base model) làm reference.
- **Dữ liệu phân tách (Data Split):** Tách 800 train / 100 held-out theo prompt chuẩn hoá (không trùng lặp bất kỳ prompt nào giữa 2 tập).
- **Biểu đồ Reward:** Phải vẽ tách riêng `rewards/chosen` và `rewards/rejected` cho cả tập huấn luyện và held-out, không chỉ vẽ margin đơn thuần.
- **Đánh giá (Evaluation):** Tối thiểu 8 prompt cố định + 50 prompt held-out; báo cáo đầy đủ win rate kèm CI 95%, sanity accuracy, length bias.
- **Gatekeeper:** Lệnh `python scripts/verify.py` phải kết thúc với mã 0 (không còn placeholder nào trong `REFLECTION.md`, đủ 4 ảnh core và file kết quả json).

## Review Focus

1. **Hiện tượng Likelihood Displacement:** Margin tăng nhưng xác suất câu `chosen` giảm (do câu `rejected` giảm nhanh hơn). Cần phát hiện qua biểu đồ reward và phân tích sâu trong §3.
2. **Thiên vị độ dài (Length Bias / Length Hack):** Dữ liệu có tỉ lệ `chosen` dài hơn cao khiến DPO có xu hướng học sinh câu dài để tăng điểm. Cần so sánh giữa win rate tổng và `length_matched_win_rate`.
3. **Rò rỉ sở thích (Preference Leakage):** Mô hình chấm cùng họ (Skywork) với mô hình gán nhãn dữ liệu có thể ưu ái DPO; cần hội đồng 2 RM (`Skywork Qwen3` + `Skywork Llama 3.2`) để lọc.
4. **Độ tin cậy của Giám khảo:** Giám khảo RM phải đạt `sanity_accuracy >= 0.8` trên bộ 12 test pairs tiếng Việt hiển nhiên.
5. **Mất phiên / Thiếu tài nguyên GPU trên Colab:** VRAM T4 16GB phù hợp cho tier T4; cần tải kết quả (`data/eval/`, `submission/screenshots/`, `adapters/dpo/*.json`) về repo trước khi hết phiên.

---

## Danh sách Task Thực hiện

### Task 1: Cài đặt và Kiểm thử DPO Loss từ đầu (NB0)
**Files:**
- Modify: `notebooks/00_dpo_loss_from_scratch.py:60-64`
- Test: `scripts/test_lab22.py`

**Interfaces:**
- Consumes: `pc`, `pr` (policy logps chosen/rejected), `rc`, `rr` (ref logps chosen/rejected), `beta: float`
- Produces: `my_dpo_loss(pc, pr, rc, rr, beta=0.1) -> torch.Tensor`

- [ ] **Step 1: Viết hàm `my_dpo_loss` hoàn chỉnh**
  Sử dụng công thức DPO:
  `loss = -torch.nn.functional.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()`
- [ ] **Step 2: Chạy kiểm thử xác nhận hàm khớp với `lab22.dpo_math.dpo_loss`**
  Chạy lệnh kiểm tra assert với đáp số tham chiếu và kiểm tra tính chất `loss = log(2)` tại bước khởi đầu.
- [ ] **Step 3: Soạn câu trả lời giải thích Likelihood Displacement cho NB0 và chuẩn bị cho REFLECTION §3**
  Giải thích vì sao margin tăng khi `chosen` log-prob giảm nếu `rejected` giảm nhanh hơn.

---

### Task 2: Kiểm tra Cấu hình, Đồng bộ Mã nguồn & Colab Notebooks
**Files:**
- Inspect: `lab22/config.py`
- Modify (nếu cần): `scripts/build_colab.py`
- Output: `colab/Lab22_DPO_T4.ipynb`

**Interfaces:**
- Consumes: Thiết lập tier `T4` (max_len=768, batch size, lr=5e-6, beta=0.1)
- Produces: File notebook Colab `colab/Lab22_DPO_T4.ipynb` cập nhật đầy đủ code mới nhất.

- [ ] **Step 1: Rà soát tham số cấu hình trong `lab22/config.py`**
  Đảm bảo `COMPUTE_TIER="T4"`, `MAX_LEN=768` (hoặc 512 nếu VRAM hẹp), `DPO_LR=5e-6`, `DPO_BETA=0.1`.
- [ ] **Step 2: Chạy `scripts/build_colab.py` để đồng bộ hoá notebook Colab**
  Chạy lệnh `python scripts/build_colab.py` để sinh file `colab/Lab22_DPO_T4.ipynb` chuẩn xác nhất từ các module trong `notebooks/`.

---

### Task 3: Huấn luyện SFT-Mini (NB1) & Gộp Trọng số Reference Model
**Files:**
- Execute: `notebooks/01_sft_mini.py` (hoặc qua cell tương ứng trên Colab T4)
- Outputs:
  - `adapters/sft-mini/` (LoRA weights & config)
  - `models/sft-merged/` (16-bit merged model làm reference cho DPO)
  - `submission/screenshots/02-sft-loss.png`

**Interfaces:**
- Consumes: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`, 1.000 mẫu `saillab/alpaca-vietnamese-cleaned`
- Produces: Reference model `models/sft-merged`, biểu đồ loss giảm dần.

- [ ] **Step 1: Huấn luyện LoRA SFT trên 1.000 mẫu tiếng Việt Alpaca**
  Chỉ tính loss trên phần trả lời (assistant response token masking).
- [ ] **Step 2: Vẽ đồ thị SFT loss và xuất file `02-sft-loss.png`**
  Xác nhận loss giảm mượt mà qua các step.
- [ ] **Step 3: Hợp nhất (merge) adapter vào mô hình nền ở định dạng 16-bit**
  Lưu vào thư mục `models/sft-merged/` để phục vụ làm reference model chuẩn cho DPO.

---

### Task 4: Chuẩn bị Dữ liệu Sở thích Tiếng Việt (NB2) & Phân tích Thiên vị Độ dài
**Files:**
- Execute: `notebooks/02_preference_data.py` (hoặc cell Colab tương ứng)
- Outputs:
  - `data/pref/train.parquet` (800 cặp)
  - `data/pref/eval.parquet` (100 cặp held-out)
  - `data/pref/stats.json`
  - `submission/screenshots/02b-pref-length.png`

**Interfaces:**
- Consumes: Dataset `sailor2/sea-ultrafeedback-onpolicy` (lọc Vietnamese)
- Produces: 2 tập dữ liệu không trùng prompt, phân tích tỷ lệ `chosen_longer_frac`.

- [ ] **Step 1: Lọc dữ liệu, chuyển đổi định dạng hội thoại và chia train/held-out theo prompt**
  Sử dụng `D.load_preference_pairs`, đảm bảo `D.assert_disjoint(train_ds, eval_ds)` vượt qua.
- [ ] **Step 2: Đánh giá 3 cặp mẫu cụ thể để khảo sát chất lượng nhãn**
  Ghi nhận nhận xét về tính tự nhiên, độ dài và mức độ ưu tiên của câu trả lời.
- [ ] **Step 3: Đo lường thiên vị độ dài và lưu đồ thị `02b-pref-length.png`**
  Tính tỷ lệ % cặp `chosen` dài hơn `rejected` (ví dụ ~60-70%) và lưu vào `stats.json`.

---

### Task 5: Huấn luyện Căn chỉnh DPO (NB3) & Giám sát Reward Curves
**Files:**
- Execute: `notebooks/03_dpo_train.py` (hoặc cell Colab tương ứng)
- Outputs:
  - `adapters/dpo/` (adapter config, adapter weights, `split.json`)
  - `adapters/dpo/dpo_metrics.json`
  - `submission/screenshots/03-dpo-reward-curves.png`

**Interfaces:**
- Consumes: `models/sft-merged/`, `data/pref/train.parquet`, `data/pref/eval.parquet`
- Produces: DPO Adapter đã căn chỉnh, log metrics và đường reward phân tích.

- [ ] **Step 1: Khởi tạo mô hình DPO từ `models/sft-merged` với LoRA mới**
  Tính toán trước `precompute_ref_log_probs=True` để tiết kiệm VRAM và tăng tốc huấn luyện.
- [ ] **Step 2: Chạy DPOTrainer với beta=0.1, lr=5e-6, đánh giá định kỳ trên held-out**
  Kiểm tra loss bước 0 xấp xỉ `log(2) ≈ 0.693` để đảm bảo reference model chuẩn xác.
- [ ] **Step 3: Xuất biểu đồ `03-dpo-reward-curves.png` và chẩn đoán tự động**
  Vẽ riêng 2 đường `rewards/chosen` và `rewards/rejected` cho cả train và held-out. Xác định nhãn chẩn đoán (`INTENDED` hoặc `LIKELIHOOD DISPLACEMENT`).
- [ ] **Step 4: Lưu `dpo_metrics.json` và kiểm tra `split.json`**

---

### Task 6: So sánh Song song Đối đầu & Chấm tự động (NB4)
**Files:**
- Execute: `notebooks/04_compare_and_eval.py` (hoặc cell Colab tương ứng)
- Outputs:
  - `data/eval/side_by_side.jsonl` (8 fixed + >= 50 held-out prompts)
  - `submission/screenshots/04-side-by-side-table.png`
  - `data/eval/judge_summary.json`

**Interfaces:**
- Consumes: `models/sft-merged/`, `adapters/dpo/`, 8 fixed prompts, eval prompts
- Produces: File so sánh, bảng so sánh trực quan, kết quả chấm tự động từ hội đồng RM.

- [ ] **Step 1: Sinh câu trả lời (Greedy Generation) cho SFT vs SFT+DPO**
  Sinh câu trả lời trên cùng tập prompt, kiểm tra độ dài trung bình ký tự.
- [ ] **Step 2: Tạo ảnh bảng so sánh 8 câu cố định `04-side-by-side-table.png`**
- [ ] **Step 3: Chạy hội đồng 2 Reward Models chấm điểm độc lập**
  Kiểm tra bộ test 12 cặp Vietnamese Sanity (`sanity_accuracy >= 0.8`), tổng hợp win rate với khoảng tin cậy 95% (bootstrap) và tỷ lệ thắng theo độ dài.
- [ ] **Step 4: Lưu kết quả vào `judge_summary.json` và kiểm tra hash SHA256**

---

### Task 7: Viết Bài Phản tư (Reflection) & Xác minh Kiểm tra Toàn diện (`make verify`)
**Files:**
- Modify: `submission/REFLECTION.md`
- Test: `scripts/verify.py`

**Interfaces:**
- Consumes: Dữ liệu thực nghiệm từ `dpo_metrics.json`, `judge_summary.json`, và các biểu đồ.
- Produces: `submission/REFLECTION.md` hoàn chỉnh (không còn placeholder), lệnh `make verify` trả về 0.

- [ ] **Step 1: Điền thông tin cấu hình và số liệu thực nghiệm vào §1 và §2**
  Cập nhật GPU, tên mô hình, thời gian train, reward gap, margin, win rate.
- [ ] **Step 2: Viết phân tích đường reward §3 (≥ 100 từ)**
  Phân tích chi tiết hướng đi của chosen/rejected trên train và held-out, margin và chẩn đoán Likelihood Displacement.
- [ ] **Step 3: Viết phân tích so sánh đối đầu SFT vs DPO §4**
  Điền bảng thống kê win rate, CI 95%, phân tích 2 ví dụ cụ thể (1 câu hữu ích, 1 câu an toàn).
- [ ] **Step 4: Viết phân tích quyết định quan trọng nhất §6 (≥ 150 từ)**
  Chọn 1 quyết định kiến trúc (ví dụ: việc đặt SFT-merged làm reference thay vì base model, hoặc lựa chọn hệ số beta=0.1) và phân tích 4 câu hỏi trọng tâm.
- [ ] **Step 5: Chạy `python scripts/verify.py` xác minh toàn bộ điều kiện nộp bài**
  Đảm bảo không còn bất kỳ lỗi nào (`✓ Core checks passed`).
