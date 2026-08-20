# Qualcomm S24 ASR benchmark — Windows runner

## Cách chạy
Yêu cầu: **Windows 10/11 64-bit**, Internet ổn định, **Python 3.11 64-bit** (khuyến nghị) và Qualcomm AI Hub API token.

1. Giải nén ZIP vào một thư mục ngắn, ví dụ `C:\qai_s24_benchmark`.
2. Double-click **`RUN_WINDOWS.bat`**.
3. Lần đầu script tự tạo `.venv` và cài dependencies.
4. Nếu chưa set `QAI_HUB_API_TOKEN`, chương trình sẽ hỏi token trong terminal. Token không được ghi vào output/evidence.
5. Nếu Windows/Internet/Qualcomm queue làm run bị dừng, chạy lại `RUN_WINDOWS.bat`; checkpoint đã xong sẽ được reuse.

## Benchmark mặc định
- FLEURS Vietnamese: **100** utterances.
- FLEURS English: **100** utterances.
- FLEURS Vietnamese noise @ 0 dB: **cùng 100 mẫu**, chạy paired clean/noisy.
- ViMD: **100 tổng**, stratified North/Central/South = 34/33/33.
- ViMedCSS test: **100** utterances.
- Models: Whisper Tiny, Whisper Small, PhoWhisper Base.
- Qualcomm microbatch: **2**, tự fallback 1 nếu payload 2 mẫu không phù hợp.
- Encoder và decoder mặc định được compile thành **hai QNN DLC riêng**, nên lượt chạy không phụ thuộc vào context-binary link.
- Mỗi inference job được tự retry tối đa **3 lần** nếu Qualcomm queue/job/download lỗi tạm thời.
- Profile latency/RAM mặc định bật và là bắt buộc để điền cột speed / nặng không; profile job được retry tối đa 3 lần. Toàn bộ 100-sample prediction vẫn chạy trên S24 NPU.

## Cái gì thực sự chạy trên S24?
Windows PC chỉ download/preprocess audio, gọi API, greedy token selection/decode và tính WER/CER. **Mọi encoder/decoder neural forward pass dùng để tạo prediction được submit qua Qualcomm AI Hub Workbench tới exact hosted `Samsung Galaxy S24`, với NPU explicitly requested.** Script abort nếu Qualcomm trả về device khác.

Thiết lập mặc định của `RUN_WINDOWS.bat` là `QAI_RUN_MODE=benchmark`, tức vẫn giữ đúng **100 mẫu cho mỗi benchmark**. Các QNN DLC và checkpoint đã hoàn thành được reuse khi chạy lại.

## Một file để nộp / upload Drive
Sau khi chạy thành công, file cần upload là:

`qai_asr_s24_benchmark\QAI_S24_BENCHMARK_SUBMISSION.zip`

ZIP này chứa:
- `FINAL_BENCHMARK_TABLE.csv` — bảng kết quả cuối.
- `RUN_EVIDENCE.json` — exact device fingerprint, QAIRT version, model/dataset revisions, QNN artifact IDs, compile/profile job IDs và representative inference job URLs.
- `QUALCOMM_JOB_LEDGER.csv` — ledger các Qualcomm inference jobs thành công được ghi trong run, gồm job ID/URL, device, graph, artifact model, batch size.
- `INFERENCE_SAMPLE_JOB_PROOF.csv` — mapping sample → encoder job ID + last decoder job ID + decoder job count.
- `SAMPLE_SELECTION.csv` — chính xác các sample đã dùng.
- `MODEL_METADATA.csv`, `s24_model_speed_memory_profile.csv`.
- `RUN_CONSOLE.log` — log chạy trên Windows (không ghi API token).
- `SHA256SUMS.txt` — hash chống sửa file evidence sau khi run.

Các Qualcomm job URL có thể mở khi đăng nhập đúng AI Hub account của nhóm để kiểm chứng job/device.

## Resume / cache
Toàn bộ QNN artifact cache, profile data và per-sample checkpoint nằm trong `qai_asr_s24_benchmark\`. Không xóa thư mục này nếu muốn resume.

- Trước khi tạo Qualcomm job, launcher chạy `pip check`, syntax check và regression preflight.
- Nếu artifact cũ có producer đã `FAILED`, script tự loại cache đó và compile lại QNN DLC thay vì dùng model hỏng.
- Profile và inference job lỗi đều được thử lại tối đa 3 lần. Benchmark chỉ báo hoàn thành khi cả 3 model có đủ encoder latency, decoder latency/token và Peak RAM; nếu vẫn lỗi, console ghi status/URL và lần chạy sau tiếp tục dùng artifact/checkpoint đã hoàn thành.
- Trước khi tạo ZIP, script kiểm tra toàn bộ ô VN/EN WER, Bắc/Trung/Nam WER, noise ΔWER, code-switch CS-WER, S24 latency và Peak RAM của cả 3 model; thiếu bất kỳ số nào thì không báo hoàn thành.
