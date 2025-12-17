# 🔧 HƯỚNG DẪN SỬA LỖI - Python 3.13+ Compatibility

## Vấn đề

`googletrans` không tương thích với Python 3.13+ do module `cgi` đã bị loại bỏ.

## Giải pháp

Đã thay thế `googletrans` bằng `deep-translator`.

## Các bước thực hiện

### 1. Gỡ cài đặt googletrans

```powershell
pip uninstall googletrans googletrans-py httpx httpcore h11 h2 -y
```

### 2. Cài đặt tất cả dependencies cần thiết (một lệnh)

```powershell
pip install deep-translator unidecode Cython phonemizer
```

### 4. Kiểm tra import

```powershell
python -c "from HmongTTS import text_to_sequence; from HmongTTS.text.symbols import symbols; print('✅ Import thành công!')"
```

### 3. Kiểm tra import

```powershell
python -c "from HmongTTS import text_to_sequence; print('✅ Import thành công!')"
```

### 4. Chạy API

```powershell
python api.py
```

## Thay đổi trong code

### 1. requirements.txt

```diff
- googletrans==4.0.0rc1
+ deep-translator>=1.11.4
```

### 2. api.py

```diff
- from googletrans import Translator
- translator = Translator()
- translation = translator.translate(text, src='hmn', dest='vi')
- vietnamese_text = translation.text

+ from deep_translator import GoogleTranslator
+ vietnamese_text = GoogleTranslator(source='auto', target='vi').translate(text)
```

### 3. Các file import fixes (relative imports)

Đã sửa các file sau để dùng relative imports:

- `HmongTTS/__init__.py`: `from .text.symbols import symbols`
- `HmongTTS/text/__init__.py`: `from . import cleaners`
- `HmongTTS/models.py`: `from . import commons, modules, attentions`
- `HmongTTS/attentions.py`: `from . import commons, modules`
- `HmongTTS/modules.py`: `from . import commons`

## Hoàn tất!

Bây giờ API đã tương thích với Python 3.13+ 🎉
