# Lab22 — kết quả lần chạy lại
Phạm Thanh Trung · 2A202602949 · Track 3 · Colab Tesla T4 · 08–09/10/2026.

## Trạng thái
- NB0: các assert đã qua; my_dpo_loss đã điền.
- NB1: SFT 1.000 mẫu, 125 bước; loss log 1,884173 → 1,283630; đã lưu adapter và SFT gộp trên Colab.
- NB2: 800 train / 100 eval, không trùng prompt; đã đọc và nhận xét 3 cặp; chosen dài hơn 65,875%.
- NB3: DPO 100 bước trên SFT gộp; held-out accuracy 69%, margin +0,0836141; đã lưu adapter và biểu đồ train/held-out.
- NB4: 8 câu cố định + 50 held-out; cả hai RM sanity 12/12, đồng thuận 94,8276%.
- REFLECTION: đủ các mục bắt buộc §1–§4, §6; §3 và §6 vượt số từ tối thiểu.
- Validator gốc tại /content/lab22: PASS, exit 0. Xem [verify_colab.txt](verify_colab.txt).
- Notebook giữ output: [Lab22-executed.ipynb](Lab22-executed.ipynb), 97 cell, không có output loại error.
- Bonus NB3b/NB5/NB6/NB7/β-sweep/API/HF Hub: chưa chạy. Repo nộp public: https://github.com/trump22/K4-L3-Track3-Day22-PhamThanhTrung-2A202602949-DPO-ORPO-Alignment. Chưa nộp link vào LMS trong lần này.

## Kết luận chất lượng
Held-out: DPO thắng 8, SFT thắng 3, hoà 39. Win rate tính hoà nửa điểm 55%, CI 95% 49–61%; ghép độ dài gần bằng 53,2609%. CI chứa 50%, chưa đủ bằng chứng DPO tốt hơn SFT. Cả hai mô hình còn lỗi thẻ tool_call và không bám đúng nguyên liệu ở h2. Đạt kiểm tra bài lab không đồng nghĩa chất lượng mô hình hoàn hảo.

## Vấn đề, nguyên nhân và xử lý
| Vấn đề | Bằng chứng / nguyên nhân | Xử lý và lý do |
|---|---|---|
| Phiên GPU trước mất file | /content không bền khi Colab thay runtime | Chạy lại NB0–NB4; tải từng nhóm kết quả về máy để giữ bằng chứng |
| Qwen RM sanity 6/12 trong phiên huấn luyện | Cùng model và hàm chấm, tiến trình Python mới không import Unsloth đạt 12/12; xác nhận môi trường ảnh hưởng, chưa xác định thư viện/bản vá cụ thể | Chấm lại cả hai RM trong tiến trình riêng, giữ bản chấm cũ; không đổi sanity, nhãn hay ngưỡng |
| MCP timeout khi chạy lâu | Thời gian chờ công cụ hết trước khi Colab chạy xong | Đọc trạng thái/output trước khi chạy tiếp, tránh chạy trùng |
| Xuất kết quả lớn làm mất kết nối | Giới hạn kích thước thông điệp WebSocket | Chuyển dữ liệu theo khối 64 KiB, đối chiếu độ dài và SHA-256 |
| Bảng so sánh tràn chữ | Nội dung dài vượt ô hình mặc định | Wrap chữ và tăng chiều cao ô; chỉ trích 250 ký tự đầu, dữ liệu đầy đủ giữ trong JSONL |
| Windows validator báo WRONG REF | Adapter ghi đường dẫn tuyệt đối của môi trường huấn luyện /content/lab22/models/sft-merged, khác đường dẫn Windows | Giữ nguyên metadata tham chiếu; chạy validator gốc trên Colab đã PASS. Không sửa validator/test hoặc đổi đường dẫn để tạo PASS giả |

## Tệp đã lưu
- data/eval/side_by_side.jsonl, judge_results_rm.json, judge_summary.json.
- adapters/dpo/adapter_config.json, dpo_metrics.json, split.json.
- data/pref/train.parquet, eval.parquet; metadata SFT và các log thực nghiệm.
- Đủ 4 ảnh trong submission/screenshots; REFLECTION.md; notebook có output; log validator.
- 27 tệp chuyển từ Colab đã đối chiếu SHA-256; ảnh bảng dựng lại cũng đã đối chiếu riêng.
- Trọng số adapter/SFT gộp nằm trên Colab; bản Windows hiện chỉ có metadata, chưa có trọng số để chạy suy luận. README không yêu cầu commit trọng số.

## Tái lập
Notebook thực thi chứa cấu hình T4, helper của lab, các bước NB0–NB4, đối chứng/chấm RM trong tiến trình riêng và validator gốc. Cell chép nguồn cuối notebook chỉ tạo file nguồn và báo cáo, không chạy bonus. Chạy bằng Colab có GPU; không cần Docker hay API trả phí. Lỗi WRONG REF của bản sao Windows vẫn tồn tại và được báo đúng phạm vi.

Test và scripts/verify.py không sửa.
