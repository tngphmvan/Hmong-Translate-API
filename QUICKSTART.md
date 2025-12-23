# Hướng dẫn Nhanh - Hmong TTS

## Cho Kaggle (Khuyến nghị)

### Bước 1: Cài đặt
```python
!pip install -q pandas openpyxl TTS torch torchaudio librosa soundfile scipy
```

### Bước 2: Sử dụng
```python
from kaggle_hmong_tts import quick_start

# Thay đổi đường dẫn file của bạn
quick_start('/kaggle/input/your-dataset/your-file.xlsx')
```

### Hoặc chi tiết hơn:
```python
from kaggle_hmong_tts import KaggleHmongTTS

processor = KaggleHmongTTS(
    output_dir='/kaggle/working/audio_output',
    tts_suffix='_tts'
)

processor.process_excel('/kaggle/input/your-dataset/file.xlsx')
```

## Cho Command Line

```bash
# Cài đặt
pip install -r requirements.txt

# Chạy
python hmong_tts_batch.py --input data.xlsx --output ./audio_output
```

## Định dạng Excel

| file_name | transcript |
|-----------|-----------|
| sample1 | Nyob zoo |
| sample2 | Ua tsaug |

## Kết quả

File WAV sẽ được tạo:
```
audio_output/
├── sample1_tts.wav
├── sample2_tts.wav
└── ...
```

## Lưu ý quan trọng

1. File Excel **phải có** 2 cột: `file_name` và `transcript`
2. GPU trên Kaggle giúp xử lý nhanh hơn
3. Mỗi file WAV khoảng 100KB - 5MB
4. Thời gian xử lý: ~5-10 giây/file

## Troubleshooting

**Lỗi module not found?**
```python
!pip install TTS pandas openpyxl
```

**Excel không đúng format?**
- Kiểm tra có đủ 2 cột: `file_name` và `transcript`
- Không có dòng trống đầu file

**Out of memory?**
- Chia nhỏ file Excel
- Bật GPU trong Kaggle settings
