# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đào Duy Hiếu

**Khoá:** K4 · mã sinh viên 2A202602651

**Tier đã chạy:** Chưa chạy tier GPU; đã kiểm tra phần CPU của repo

**Ngày:** 2026-10-08

> Báo cáo này phân biệt cấu hình dự kiến, kiểm tra phần mềm và thí nghiệm mô hình. Notebook T4 trong workspace hiện không có execution count hoặc output đã lưu. Không có bằng chứng mô hình SFT/DPO hay giám khảo đã chạy.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| Máy cục bộ | Intel Core i7-13650HX — 14 lõi (6P+8E), 20 luồng logic, turbo tối đa 4.90 GHz theo thông số Intel; RAM 23.8 GiB; Windows 11 build 26200. [Thông số CPU](https://www.intel.com/content/www/us/en/products/sku/232101/intel-core-i713650hx-processor-24m-cache-up-to-4-90-ghz/specifications.html) |
| GPU / VRAM | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM (~8.0 GiB), driver 610.62; còn 5,935 MiB (~5.8 GiB) tại thời điểm kiểm tra. Dưới mức 12 GB ghi trong hướng dẫn lab. |
| Dung lượng đĩa D: | 1,000,186,310,656 byte tổng; 146,095,730,688 byte trống tại thời điểm kiểm tra. |
| Môi trường Python | PyTorch chưa được cài tại thời điểm kiểm tra; chưa xác nhận CUDA qua PyTorch. |
| Mô hình gốc | Chưa nạp/chạy. Notebook cấu hình mặc định Qwen3-4B-Instruct-2507 4-bit. |
| Dữ liệu SFT | Chưa tải hoặc huấn luyện trong phiên có bằng chứng. |
| Dữ liệu sở thích | Workspace có Parquet 800 train / 100 eval, prompt không trùng. Mẫu đã kiểm tra là văn bản tổng hợp dạng “Câu hỏi thực hành 0”, nên không được xem là dữ liệu thật từ nguồn đã nêu trong `stats.json`. |
| Chosen dài hơn rejected (NB2) | Chưa có thống kê hợp lệ từ tokenizer. `stats.json` ghi 65.2%, nhưng dữ liệu mẫu tổng hợp khiến con số này không được dùng làm kết quả thí nghiệm. |
| DPO: β / tốc độ học (lr) / số epoch | Notebook đặt 0.1 / 5e-6 / 1; chưa chạy. |
| Giám khảo | Chưa chạy; không có sanity accuracy. |
| Chi phí | Chưa thuê GPU. Không có thông tin xác nhận về khoản phí Colab. |

Đã chạy bộ kiểm thử cục bộ bằng `python -m pytest -p no:cacheprovider scripts/ -q`: **5 passed, 1 skipped**. Đây là kiểm tra mã CPU, không phải bằng chứng các cell NB0–NB4 đã chạy trong notebook. Sau khi điền báo cáo này, `python scripts/verify.py` vẫn báo thiếu metrics DPO, kết quả judge và bốn ảnh bắt buộc.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Chưa chạy |
| VRAM cao nhất | Chưa đo |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | Chưa có số liệu |
| Độ chính xác reward trên held-out | Chưa có số liệu |
| Margin trên held-out | Chưa có số liệu |
| Chẩn đoán tự động (`diagnosis`) | Chưa có |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | Chưa sinh câu trả lời |

---

## 3. Đọc đường reward

Hiện chưa có đường reward từ một lần huấn luyện DPO thật để phân tích. Notebook chưa lưu output và workspace không có `adapters/dpo/dpo_metrics.json` từ trainer. Vì vậy chưa thể kết luận `rewards/chosen` tăng hay giảm, `rewards/rejected` thay đổi ra sao, margin tăng nhờ phía nào, hoặc held-out có đi cùng hướng với train hay không. Các biểu đồ trong `submission/mock-artifacts/` là dữ liệu tổng hợp để minh họa định dạng; chúng không phải log của mô hình và không được dùng để suy luận về likelihood displacement hay overfit. Phần đã kiểm tra được là bộ test CPU của repo chạy với 5 test qua và 1 test bị skip; việc đó chỉ xác nhận một số hàm phần mềm, không kiểm tra hành vi của mô hình đã fine-tune. Bước tiếp theo để hoàn tất mục này là chạy NB1 và NB3 trên GPU, giữ lại `dpo_metrics.json` cùng biểu đồ có train và held-out, rồi mô tả xu hướng quan sát được từ chính các file đó.

---

## 4. So sánh SFT vs SFT+DPO

Chưa có đầu ra từ hai mô hình hoặc kết quả giám khảo. Các tệp minh họa trong `submission/mock-artifacts/` không được tính là kết quả đánh giá.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | Chưa chấm | — | — | — | — | — | — |
| hữu ích — helpfulness (4) | Chưa chấm | — | — | — | — | — | — |
| an toàn — safety (4) | Chưa chấm | — | — | — | — | — | — |

Giám khảo: chưa chạy · sanity accuracy: chưa đo · score-length correlation / position consistency: chưa đo.

Chưa thể chọn ví dụ thắng/thua thật vì NB4 chưa sinh câu trả lời. Khi chạy lại, cần lưu 8 câu cố định và ít nhất 50 prompt held-out, sau đó dùng kết quả giám khảo để hoàn thiện bảng và phân tích. Không dùng các verdict tạo sẵn trong mock bundle làm ví dụ thực nghiệm.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Chưa chạy bonus β-sweep; không có số liệu thực nghiệm để so sánh.

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | Chưa chạy | Chưa chạy | — | — |
| 0.1 | Chưa chạy | Chưa chạy | — | — |
| 0.5 | Chưa chạy | Chưa chạy | — | — |

---

## 6. Một quyết định quan trọng nhất

Quyết định quan trọng nhất ở giai đoạn này là không coi GPU laptop RTX 4060 với 8 GB VRAM như thể nó tương đương T4 16 GB của cấu hình lab. Máy có i7-13650HX với 14 lõi (6 lõi hiệu năng và 8 lõi tiết kiệm điện), 20 luồng logic, turbo tối đa 4.90 GHz theo thông số nhà sản xuất và 23.8 GiB RAM, nhưng GPU thấp hơn mức 12 GB tối thiểu nêu trong hướng dẫn; tại lúc kiểm tra còn khoảng 5.8 GiB VRAM trống. Phương án khác là mượn hoặc thuê GPU đủ bộ nhớ, hoặc chỉnh cấu hình để thử mô hình nhỏ hơn/giới hạn độ dài. Phương án thứ hai có thể cần ít VRAM hơn nhưng sẽ tạo thành một thí nghiệm khác với cấu hình mặc định, nên phải ghi rõ thay đổi. Ngoài ra, PyTorch chưa có trong môi trường Python hiện tại và tài khoản Colab gặp lỗi theo tình trạng đã báo. Vì notebook không lưu execution count hay output, chưa có cơ sở để nói rằng huấn luyện đã khởi động rồi chạy lâu mới lỗi; phần cứng và môi trường chỉ giải thích vì sao khó tiếp tục. Kiểm thử CPU của repo qua 5 test và skip 1 test, nhưng không xác nhận chất lượng căn chỉnh. Nếu làm lại, tôi sẽ cài đúng môi trường trong runtime GPU đủ VRAM, kiểm tra CUDA trước, chạy NB0 rồi NB1–NB4, lưu log/artifact từng bước và chỉ điền số liệu report từ các file thật. Nếu không có GPU trước hạn nộp, tôi sẽ nộp phần đã xác minh cùng giới hạn này hoặc xin giảng viên phương án thay thế.

---

## 7. Bộ đo chuẩn (bonus NB6)

Chưa chạy NB6; không có điểm benchmark để báo cáo.

## 8. Biến thể loss (bonus NB3b)

Chưa chạy NB3b; không có kết quả so sánh các biến thể loss.

## 9. GRPO (bonus NB7)

Chưa chạy NB7; không có số liệu độ chính xác trước/sau hoặc reward GRPO.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss
- [ ] NB5 — GGUF SFT+DPO
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng hai họ mô hình
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình
- [ ] `BONUS-CHALLENGE.md`

## Điều bất ngờ nhất

Các file Parquet có đúng số dòng và chia prompt không trùng, nhưng mẫu nội dung kiểm tra được là văn bản tổng hợp chứ không giống cặp dữ liệu gốc có provenance. Vì vậy kiểm tra được định dạng và cách chia không đồng nghĩa đã hoàn tất NB2 trên dữ liệu thật.
