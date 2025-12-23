# Script cài đặt tất cả dependencies cho Hmong Translation API
# Chạy: .\install_dependencies.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Cài đặt Dependencies" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kích hoạt virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Kích hoạt virtual environment..." -ForegroundColor Green
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  Virtual environment chưa tồn tại. Tạo mới..." -ForegroundColor Yellow
    python -m venv venv
    & ".\venv\Scripts\Activate.ps1"
}

# Gỡ cài đặt googletrans cũ
Write-Host ""
Write-Host "🗑️  Gỡ cài đặt googletrans cũ (nếu có)..." -ForegroundColor Yellow
pip uninstall googletrans googletrans-py httpx httpcore h11 h2 -y 2>$null

# Cài đặt dependencies chính
Write-Host ""
Write-Host "📦 Cài đặt dependencies chính..." -ForegroundColor Green
pip install --upgrade pip

Write-Host ""
Write-Host "📦 Cài đặt FastAPI và server..." -ForegroundColor Green
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 python-multipart==0.0.6

Write-Host ""
Write-Host "📦 Cài đặt Google Translator..." -ForegroundColor Green
pip install deep-translator>=1.11.4

Write-Host ""
Write-Host "📦 Cài đặt PyTorch và Transformers..." -ForegroundColor Green
pip install torch>=2.0.0 transformers>=4.35.0 accelerate>=0.24.0

Write-Host ""
Write-Host "📦 Cài đặt audio processing..." -ForegroundColor Green
pip install numpy>=1.24.0 scipy>=1.11.0 librosa>=0.10.0 soundfile>=0.12.0

Write-Host ""
Write-Host "📦 Cài đặt VITS TTS dependencies..." -ForegroundColor Green
pip install unidecode>=1.3.0 Cython>=3.0.0 phonemizer>=3.2.0

Write-Host ""
Write-Host "📦 Cài đặt utilities..." -ForegroundColor Green
pip install pydantic>=2.0.0

# Build monotonic_align
Write-Host ""
Write-Host "🔨 Build monotonic_align module..." -ForegroundColor Green
if (Test-Path "HmongTTS\monotonic_align") {
    Push-Location "HmongTTS\monotonic_align"
    python setup.py build_ext --inplace
    Pop-Location
    Write-Host "✅ Monotonic align đã được build" -ForegroundColor Green
} else {
    Write-Host "⚠️  Không tìm thấy thư mục monotonic_align" -ForegroundColor Yellow
}

# Test import
Write-Host ""
Write-Host "🧪 Test import..." -ForegroundColor Green
python -c "from HmongTTS import text_to_sequence; from HmongTTS.text.symbols import symbols; print('✅ HmongTTS import thành công!')"

if ($LASTEXITCODE -eq 0) {
    python -c "from deep_translator import GoogleTranslator; print('✅ Deep Translator import thành công!')"
}

if ($LASTEXITCODE -eq 0) {
    python -c "from fastapi import FastAPI; print('✅ FastAPI import thành công!')"
}

# Hoàn tất
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✅ Hoàn tất cài đặt!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Bây giờ bạn có thể chạy API:" -ForegroundColor Yellow
Write-Host "  python api.py" -ForegroundColor Cyan
Write-Host ""
