# Quick start script for the API
# Usage: .\start_api.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Hmong-Vietnamese Translation API" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check virtual environment
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "⚠️  Virtual environment chưa được tạo!" -ForegroundColor Yellow
    Write-Host "Đang tạo virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "🔧 Kích hoạt virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

# Check dependencies
Write-Host "📦 Kiểm tra dependencies..." -ForegroundColor Green
$requirementsInstalled = $true

try {
    python -c "import fastapi" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $requirementsInstalled = $false
    }
} catch {
    $requirementsInstalled = $false
}

if (-not $requirementsInstalled) {
    Write-Host "⚠️  Dependencies chưa được cài đặt!" -ForegroundColor Yellow
    Write-Host "Đang cài đặt dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Check whether monotonic_align has been built
$monotonicBuildPath = "HmongTTS\monotonic_align\build"
if (-not (Test-Path $monotonicBuildPath)) {
    Write-Host "🔨 Build monotonic_align..." -ForegroundColor Green
    Push-Location "HmongTTS\monotonic_align"
    python setup.py build_ext --inplace
    Pop-Location
}

# Start the API
Write-Host ""
Write-Host "🚀 Khởi động API server..." -ForegroundColor Green
Write-Host "   URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Nhấn Ctrl+C để dừng server" -ForegroundColor Yellow
Write-Host ""

python api.py
