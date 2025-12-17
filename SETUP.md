# 🚀 SETUP NHANH - 2 LỆNH

## ⚡ Cài đặt (chạy 1 lần)

```powershell
.\install_dependencies.ps1
```

**Hoặc cài thủ công:**

```powershell
# Kích hoạt venv
.\venv\Scripts\Activate.ps1

# Gỡ googletrans cũ
pip uninstall googletrans googletrans-py httpx httpcore h11 h2 -y

# Cài tất cả dependencies
pip install deep-translator unidecode Cython phonemizer fastapi uvicorn[standard] python-multipart torch transformers accelerate numpy scipy librosa soundfile pydantic

# Build monotonic_align
cd HmongTTS\monotonic_align
python setup.py build_ext --inplace
cd ..\..
```

## ▶️ Chạy API

```powershell
python api.py
```

Truy cập: **http://localhost:8000/docs**

---

## 📝 2 API có sẵn:

1. **Mông → Việt**: `POST /api/hmong-to-vietnamese` - Upload audio Mông, nhận text Việt
2. **Việt → Mông**: `POST /api/vietnamese-to-hmong` - Upload audio Việt, nhận audio Mông

## 🐛 Nếu gặp lỗi:

```powershell
# Kiểm tra import
python -c "from HmongTTS import text_to_sequence; print('OK')"

# Nếu lỗi "No module named 'unidecode'":
pip install unidecode

# Nếu lỗi "No module named 'deep_translator'":
pip install deep-translator

# Nếu lỗi monotonic_align:
cd HmongTTS\monotonic_align
python setup.py build_ext --inplace
cd ..\..
```

## 📚 Tài liệu chi tiết:

- [README_API.md](README_API.md) - Hướng dẫn sử dụng API đầy đủ
- [QUICKSTART.md](QUICKSTART.md) - Hướng dẫn nhanh
- [FIX_PYTHON313.md](FIX_PYTHON313.md) - Sửa lỗi Python 3.13+

---

**Yêu cầu:** Python 3.8+ | RAM: 8GB+ | GPU: Tùy chọn (khuyến nghị)
