# Hmong Text-to-Speech Batch Processing

Công cụ xử lý hàng loạt chuyển văn bản tiếng Hmong thành giọng nói (Text-to-Speech) từ file Excel.

## Tính năng

- ✅ Đọc file Excel với 2 cột: `file_name` và `transcript`
- ✅ Chuyển đổi văn bản tiếng Hmong sang file âm thanh WAV
- ✅ Tự động đặt tên file theo cột `file_name` + hậu tố TTS
- ✅ Lưu tất cả file WAV vào thư mục đầu ra chỉ định
- ✅ Tối ưu hóa cho môi trường Kaggle
- ✅ Hỗ trợ GPU để xử lý nhanh hơn
- ✅ Logging chi tiết và báo cáo tiến trình

## Cài đặt

### Trên Kaggle

```python
# Chạy lệnh này trong Kaggle notebook cell
!pip install -q pandas openpyxl TTS torch torchaudio librosa soundfile scipy
```

### Trên máy local

```bash
pip install -r requirements.txt
```

## Cách sử dụng

### Phương pháp 1: Sử dụng script Kaggle (Khuyến nghị cho Kaggle)

1. Upload file `kaggle_hmong_tts.py` và file Excel vào Kaggle
2. Trong Kaggle notebook:

```python
# Cài đặt dependencies
!pip install -q pandas openpyxl TTS torch torchaudio librosa soundfile scipy

# Import và chạy
from kaggle_hmong_tts import quick_start

# Xử lý file Excel
quick_start('/kaggle/input/your-dataset/your-file.xlsx')
```

**Hoặc sử dụng chi tiết hơn:**

```python
from kaggle_hmong_tts import KaggleHmongTTS

# Khởi tạo processor
processor = KaggleHmongTTS(
    output_dir='/kaggle/working/audio_output',
    tts_suffix='_tts'
)

# Xử lý file
processor.process_excel('/kaggle/input/your-dataset/transcripts.xlsx')
```

### Phương pháp 2: Sử dụng command line script

```bash
python hmong_tts_batch.py --input data.xlsx --output ./audio_output
```

**Các tùy chọn:**

```bash
python hmong_tts_batch.py \
    --input data.xlsx \
    --output ./audio_output \
    --suffix _hmong_tts \
    --verbose
```

**Tham số:**
- `--input, -i`: Đường dẫn đến file Excel đầu vào (bắt buộc)
- `--output, -o`: Thư mục lưu file WAV đầu ra (bắt buộc)
- `--suffix, -s`: Hậu tố thêm vào tên file (mặc định: `_tts`)
- `--verbose, -v`: Hiển thị log chi tiết

## Định dạng file Excel

File Excel cần có **2 cột bắt buộc**:

| file_name | transcript |
|-----------|-----------|
| hmong_sample_1 | Nyob zoo kuv lub npe yog Maria |
| hmong_sample_2 | Koj nyob li cas |
| hmong_sample_3 | Kuv nyob zoo Ua tsaug |

- **file_name**: Tên file đầu ra (không cần đuôi .wav)
- **transcript**: Văn bản tiếng Hmong cần chuyển thành giọng nói

Xem file mẫu: `example_input.csv` hoặc tạo file Excel với định dạng tương tự.

## Kết quả đầu ra

Các file WAV sẽ được tạo với định dạng:
```
audio_output/
├── hmong_sample_1_tts.wav
├── hmong_sample_2_tts.wav
├── hmong_sample_3_tts.wav
└── ...
```

## Thư viện sử dụng

- **pandas**: Đọc file Excel
- **openpyxl**: Hỗ trợ định dạng .xlsx
- **TTS (Coqui TTS)**: Engine Text-to-Speech
- **torch/torchaudio**: Deep learning models
- **librosa/soundfile**: Xử lý audio
- **scipy**: Tính toán khoa học

## Ví dụ đầy đủ cho Kaggle

```python
# ============================================================================
# CELL 1: Cài đặt dependencies
# ============================================================================
!pip install -q pandas openpyxl TTS torch torchaudio librosa soundfile scipy

# ============================================================================
# CELL 2: Upload và xử lý
# ============================================================================
from kaggle_hmong_tts import KaggleHmongTTS

# Khởi tạo
processor = KaggleHmongTTS(
    output_dir='/kaggle/working/hmong_audio',
    tts_suffix='_tts'
)

# Xử lý file Excel (thay đổi đường dẫn theo file của bạn)
processor.process_excel('/kaggle/input/hmong-data/transcripts.xlsx')

# ============================================================================
# CELL 3: Kiểm tra kết quả
# ============================================================================
import os
from pathlib import Path

output_dir = Path('/kaggle/working/hmong_audio')
wav_files = list(output_dir.glob('*.wav'))

print(f"Đã tạo {len(wav_files)} file WAV:")
for f in wav_files[:5]:
    print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
```

## Lưu ý

1. **Mô hình TTS**: Script sử dụng Coqui TTS với mô hình đa ngôn ngữ. Nếu có mô hình Hmong chuyên biệt, có thể thay đổi trong code.

2. **GPU trên Kaggle**: Script tự động phát hiện và sử dụng GPU nếu có sẵn để tăng tốc độ xử lý.

3. **Dung lượng file**: Mỗi file WAV có thể có dung lượng 100KB - 5MB tùy thuộc vào độ dài văn bản.

4. **Giới hạn**: Coqui TTS có thể không hỗ trợ hoàn hảo tiếng Hmong. Chất lượng giọng nói có thể khác nhau tùy vào mô hình được chọn.

## Khắc phục sự cố

**Lỗi: Module 'TTS' not found**
```bash
pip install TTS
```

**Lỗi: Module 'openpyxl' not found**
```bash
pip install openpyxl
```

**Lỗi: File Excel không đúng định dạng**
- Đảm bảo file có 2 cột: `file_name` và `transcript`
- Kiểm tra không có dòng trống ở đầu file

**Lỗi: Out of memory trên Kaggle**
- Giảm số lượng file xử lý mỗi lần
- Đảm bảo đã bật GPU trong Kaggle settings

## Giấy phép

MIT License

## Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.
