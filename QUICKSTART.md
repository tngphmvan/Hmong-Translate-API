# 🚀 HƯỚNG DẪN NHANH - HMONG-VIETNAMESE TRANSLATION API

## Cài đặt và Chạy (3 bước)

### Bước 1: Cài đặt dependencies

```powershell
# Kích hoạt virtual environment
.\venv\Scripts\Activate.ps1

# Cài đặt packages
pip install -r requirements.txt

# Build monotonic_align
cd HmongTTS\monotonic_align
python setup.py build_ext --inplace
cd ..\..
```

### Bước 2: Khởi động API

```powershell
# Cách 1: Chạy trực tiếp
python api.py

# Cách 2: Dùng script tự động
.\start_api.ps1
```

Server sẽ chạy tại: **http://localhost:8000**

### Bước 3: Test API

```powershell
# Mở trình duyệt và truy cập:
http://localhost:8000/docs

# Hoặc test bằng Python:
python test_api.py
```

## 📋 2 API chính

### 1️⃣ API: Mông → Việt

**Endpoint:** `POST /api/hmong-to-vietnamese`

**Chức năng:** Nhận audio tiếng Mông → Trả về text tiếng Việt

**Cách dùng (Python):**

```python
import requests

with open("hmong_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/hmong-to-vietnamese",
        files={"audio": f}
    )
    result = response.json()
    print(result["vietnamese_text"])
```

**Cách dùng (curl):**

```bash
curl -X POST "http://localhost:8000/api/hmong-to-vietnamese" \
  -F "audio=@hmong_audio.wav"
```

**Output:**

```json
{
  "hmong_text": "Nyob zoo",
  "vietnamese_text": "Xin chào",
  "success": true
}
```

---

### 2️⃣ API: Việt → Mông

**Endpoint:** `POST /api/vietnamese-to-hmong`

**Chức năng:** Nhận audio tiếng Việt → Trả về audio tiếng Mông

**Cách dùng (Python):**

```python
import requests

with open("vietnamese_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/vietnamese-to-hmong",
        files={"audio": f}
    )

    # Lưu file audio kết quả
    with open("output_hmong.wav", "wb") as out:
        out.write(response.content)

    # Xem text từ headers
    print("Việt:", response.headers.get('X-Vietnamese-Text'))
    print("Mông:", response.headers.get('X-Hmong-Text'))
```

**Cách dùng (curl):**

```bash
curl -X POST "http://localhost:8000/api/vietnamese-to-hmong" \
  -F "audio=@vietnamese_audio.wav" \
  --output output_hmong.wav
```

**Output:** File WAV audio tiếng Mông

---

## 🧪 Test nhanh

```powershell
# 1. Kiểm tra API hoạt động
curl http://localhost:8000/health

# 2. Xem thông tin API
curl http://localhost:8000/

# 3. Test với file audio
python test_api.py hmong_sample.wav vietnamese_sample.wav
```

## 📊 Quy trình xử lý

### API 1: Mông → Việt

```
Audio Mông → Whisper ASR → Text Mông → Google Translate → Text Việt
```

### API 2: Việt → Mông

```
Audio Việt → Whisper ASR → Text Việt → Google Translate → Text Mông → VITS TTS → Audio Mông
```

## ⚙️ Models sử dụng

| Chức năng  | Model                 | Mục đích                       |
| ---------- | --------------------- | ------------------------------ |
| ASR Mông   | Whisper Large (Hmong) | Nhận dạng giọng nói tiếng Mông |
| ASR Việt   | Whisper Base          | Nhận dạng giọng nói tiếng Việt |
| Dịch thuật | Google Translate      | Dịch Mông ↔ Việt               |
| TTS Mông   | VITS Custom           | Tạo giọng nói tiếng Mông       |

## 🔧 Cấu hình

### Thay đổi device (CPU/GPU)

Mở `api.py` và sửa:

```python
device_asr = "cuda:0"  # hoặc "cpu"
device_tts = "cpu"     # khuyến nghị CPU cho TTS
```

### Điều chỉnh chất lượng TTS

```python
noise_scale=0.667,      # Giảm để giọng nói rõ hơn
noise_scale_w=0.8,      # Điều chỉnh biến âm
length_scale=1          # Tốc độ đọc (>1 = chậm, <1 = nhanh)
```

## 🐛 Xử lý lỗi thường gặp

### 1. Module 'monotonic_align' không tìm thấy

```powershell
cd HmongTTS\monotonic_align
python setup.py build_ext --inplace
cd ..\..
```

### 2. API không khởi động

- Kiểm tra port 8000 có bị chiếm không
- Xem log lỗi trong terminal
- Đảm bảo đã cài đủ dependencies

### 3. Lỗi khi dịch

- Kiểm tra file audio có đúng định dạng
- File audio phải có nội dung rõ ràng
- Kiểm tra kết nối internet (Google Translate cần internet)

## 📝 Format audio hỗ trợ

- ✅ WAV (khuyến nghị)
- ✅ MP3
- ✅ M4A
- ✅ FLAC
- ✅ OGG

## 💡 Tips

1. **Tốc độ xử lý:**

   - GPU: Nhanh hơn 5-10 lần cho ASR
   - CPU: Đủ dùng cho TTS

2. **Chất lượng audio:**

   - Nên dùng WAV 16kHz hoặc 22.05kHz
   - Mono channel tốt hơn stereo
   - File ngắn (~5-30 giây) xử lý nhanh nhất

3. **Bảo mật:**
   - Thêm authentication nếu deploy public
   - Giới hạn kích thước file upload
   - Rate limiting cho API

## 📚 Tài liệu thêm

- **API Documentation:** http://localhost:8000/docs
- **README chi tiết:** [README_API.md](README_API.md)
- **Code mẫu:** [test_api.py](test_api.py)

## 📞 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra logs trong terminal
2. Xem file `README_API.md` để biết thêm chi tiết
3. Test từng bước với `test_api.py`

---

**Chúc bạn sử dụng thành công! 🎉**
