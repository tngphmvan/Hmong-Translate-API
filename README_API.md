# Hmong-Vietnamese Translation API

API dịch thuật hai chiều giữa tiếng Mông và tiếng Việt với khả năng nhận diện giọng nói (ASR) và tổng hợp giọng nói (TTS).

## 🌟 Tính năng

### API 1: Mông → Việt

- **Endpoint**: `POST /api/hmong-to-vietnamese`
- **Input**: File âm thanh tiếng Mông (WAV, MP3, M4A, etc.)
- **Output**: JSON với văn bản tiếng Mông và tiếng Việt
- **Quy trình**:
  1. Nhận file âm thanh tiếng Mông
  2. Sử dụng Whisper ASR để chuyển thành text
  3. Dịch sang tiếng Việt bằng Google Translate
  4. Trả về kết quả JSON

### API 2: Việt → Mông

- **Endpoint**: `POST /api/vietnamese-to-hmong`
- **Input**: File âm thanh tiếng Việt (WAV, MP3, M4A, etc.)
- **Output**: File âm thanh tiếng Mông (WAV)
- **Quy trình**:
  1. Nhận file âm thanh tiếng Việt
  2. Sử dụng Whisper ASR để chuyển thành text
  3. Dịch sang tiếng Mông bằng Google Translate
  4. Sử dụng VITS TTS để tạo audio tiếng Mông
  5. Trả về file audio

## 📦 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd "Hmong Translate API"
```

### 2. Tạo môi trường ảo

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Cài đặt dependencies

```powershell
pip install -r requirements.txt
```

### 4. Build monotonic_align (cho VITS TTS)

```powershell
cd HmongTTS\monotonic_align
python setup.py build_ext --inplace
cd ..\..
```

## 🚀 Chạy API

### Khởi động server

```powershell
python api.py
```

Server sẽ chạy tại: `http://localhost:8000`

### API Documentation

Sau khi khởi động server, truy cập:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📝 Sử dụng API

### Cách 1: Sử dụng curl

#### API 1: Mông → Việt

```bash
curl -X POST "http://localhost:8000/api/hmong-to-vietnamese" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@hmong_audio.wav"
```

**Response:**

```json
{
  "hmong_text": "Nyob zoo",
  "vietnamese_text": "Xin chào",
  "success": true,
  "message": "Dịch thành công"
}
```

#### API 2: Việt → Mông

```bash
curl -X POST "http://localhost:8000/api/vietnamese-to-hmong" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@vietnamese_audio.wav" \
  --output hmong_output.wav
```

**Response**: File âm thanh WAV tiếng Mông

### Cách 2: Sử dụng Python

```python
import requests

# API 1: Mông → Việt
with open("hmong_audio.wav", "rb") as f:
    files = {"audio": f}
    response = requests.post(
        "http://localhost:8000/api/hmong-to-vietnamese",
        files=files
    )
    result = response.json()
    print(f"Tiếng Mông: {result['hmong_text']}")
    print(f"Tiếng Việt: {result['vietnamese_text']}")

# API 2: Việt → Mông
with open("vietnamese_audio.wav", "rb") as f:
    files = {"audio": f}
    response = requests.post(
        "http://localhost:8000/api/vietnamese-to-hmong",
        files=files
    )
    with open("output_hmong.wav", "wb") as out:
        out.write(response.content)

    # Lấy thông tin từ headers
    print(f"Tiếng Việt: {response.headers.get('X-Vietnamese-Text')}")
    print(f"Tiếng Mông: {response.headers.get('X-Hmong-Text')}")
```

### Cách 3: Sử dụng Postman

1. Mở Postman
2. Tạo request mới với method `POST`
3. URL: `http://localhost:8000/api/hmong-to-vietnamese` hoặc `/api/vietnamese-to-hmong`
4. Trong tab **Body**, chọn **form-data**
5. Thêm key `audio` với type `File` và chọn file audio
6. Click **Send**

## 🏗️ Cấu trúc Project

```
Hmong Translate API/
├── api.py                      # Main API file
├── requirements.txt            # Python dependencies
├── README_API.md              # API documentation
├── hmongtts.py                # ASR script (Whisper)
├── HmongTTS/                  # VITS TTS module
│   ├── app.py                 # Gradio demo
│   ├── hmong.json             # TTS config
│   ├── G_60000.pth            # TTS model weights
│   ├── models.py              # Model architecture
│   ├── utils.py               # Utilities
│   ├── symbols.py             # Phoneme symbols
│   └── monotonic_align/       # Alignment module
└── venv/                      # Virtual environment
```

## 🔧 Cấu hình

### Models sử dụng

- **ASR (Mông)**: `Pakorn2112/whisper-model-large-hmong`
- **ASR (Việt)**: `openai/whisper-base`
- **Dịch thuật**: Google Translate API
- **TTS (Mông)**: VITS custom model (`G_60000.pth`)

### Yêu cầu hệ thống

- Python 3.8+
- RAM: Tối thiểu 8GB (khuyến nghị 16GB)
- GPU: Tùy chọn (khuyến nghị cho tốc độ nhanh hơn)

### Tùy chỉnh

Chỉnh sửa các tham số trong `api.py`:

- `device_asr`: Thiết bị cho ASR ("cuda:0" hoặc "cpu")
- `device_tts`: Thiết bị cho TTS (khuyến nghị "cpu")
- `chunk_length_s`: Độ dài chunk cho Whisper (mặc định: 30s)
- `noise_scale`, `length_scale`: Tham số TTS

## 🧪 Testing

### Health Check

```bash
curl http://localhost:8000/health
```

### API Info

```bash
curl http://localhost:8000/
```

## 🐛 Xử lý lỗi

### Lỗi thường gặp

1. **"Không nhận dạng được văn bản từ audio"**

   - Kiểm tra file audio có đúng định dạng
   - Đảm bảo audio có nội dung rõ ràng

2. **"Text tiếng Mông không hợp lệ sau khi làm sạch"**

   - Google Translate có thể trả về ký tự không hợp lệ
   - Kiểm tra lại văn bản nguồn

3. **Model loading errors**
   - Đảm bảo đã cài đặt đầy đủ dependencies
   - Kiểm tra đường dẫn đến model files

## 📊 Performance

- **API 1 (Mông → Việt)**: ~5-10 giây/request (tùy độ dài audio)
- **API 2 (Việt → Mông)**: ~10-15 giây/request (bao gồm TTS)

Thời gian xử lý phụ thuộc vào:

- Độ dài file audio
- Thiết bị (CPU/GPU)
- Tốc độ Google Translate API

## 📄 License

[Thêm license của bạn ở đây]

## 🤝 Contributing

Đóng góp luôn được chào đón! Vui lòng:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📧 Contact

[Thêm thông tin liên hệ của bạn]
