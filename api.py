"""
Hmong-Vietnamese Translation API
API dịch thuật giữa tiếng Mông và tiếng Việt
"""
from groq import Groq
import HmongTTS.commons as commons
import HmongTTS.utils as utils
from HmongTTS.models import SynthesizerTrn
from HmongTTS.text.symbols import symbols
from HmongTTS import text_to_sequence
import sherpa_onnx
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import os
import sys
import io
import base64
import torch
import tempfile
import numpy as np
from urllib.parse import quote
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import wave
import soundfile as sf
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
load_dotenv()
# Groq API for Translation

# Khởi tạo Groq client
groq_client = Groq()


def translate_with_groq(text: str, source_lang: str, target_lang: str) -> str:
    """Dịch văn bản sử dụng Groq API"""
    if source_lang == "hmn" and target_lang == "vi":
        system_prompt = "Translate from Hmong to Vietnamese. Just translate, no explanation."
    elif source_lang == "vi" and target_lang == "hmn":
        system_prompt = "Translate from Vietnamese to Hmong. Just translate, no explanation."
    else:
        system_prompt = f"Translate from {source_lang} to {target_lang}. Just translate, no explanation."

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content.strip()

# Import cho Whisper (ASR - Mông sang Text)

# Import cho Sherpa-ONNX (Zipformer - Vietnamese ASR)

# Import cho VITS TTS (Text Mông sang Audio)


# ==================== CẤU HÌNH ====================
app = FastAPI(
    title="Hmong-Vietnamese Translation API",
    description="API dịch giữa tiếng Mông và tiếng Việt với ASR và TTS",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== KHỞI TẠO MODELS ====================
print("🔧 Đang khởi tạo các models...")

# 1. Google Translator
# deep-translator không cần khởi tạo, sẽ tạo instance khi dùng

# 2. Whisper ASR (Hmong Speech -> Text)
WHISPER_REPO = "Pakorn2112/whisper-model-large-hmong"
WHISPER_SUBFOLDER = "SingleSpeech"
device_asr = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"📡 Đang tải Whisper model cho ASR (device: {device_asr})...")
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_REPO,
    subfolder=WHISPER_SUBFOLDER,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)
asr_model.to(device_asr)

asr_processor = AutoProcessor.from_pretrained(
    WHISPER_REPO, subfolder=WHISPER_SUBFOLDER)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=asr_processor.tokenizer,
    feature_extractor=asr_processor.feature_extractor,
    chunk_length_s=30,
    device=device_asr,
)

# 3. Zipformer ASR (Vietnamese Speech -> Text)
ZIPFORMER_REPO = "hynt/Zipformer-30M-RNNT-6000h"
ZIPFORMER_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "zipformer_vi_model")
os.makedirs(ZIPFORMER_MODEL_DIR, exist_ok=True)

print("📡 Đang tải Zipformer model cho Vietnamese ASR...")

# Download các file model cần thiết từ Hugging Face
zipformer_files = {
    "encoder": "encoder-epoch-20-avg-10.onnx",
    "decoder": "decoder-epoch-20-avg-10.onnx",
    "joiner": "joiner-epoch-20-avg-10.onnx",
    "tokens": "config.json"  # File này thực tế là tokens.txt format
}

for key, filename in zipformer_files.items():
    local_path = os.path.join(ZIPFORMER_MODEL_DIR, filename)
    if not os.path.exists(local_path):
        print(f"  Đang tải {filename}...")
        hf_hub_download(
            repo_id=ZIPFORMER_REPO,
            filename=filename,
            local_dir=ZIPFORMER_MODEL_DIR,
            local_dir_use_symlinks=False
        )

# config.json thực tế là tokens.txt format (token id per line)
# Sử dụng trực tiếp làm tokens file
tokens_path = os.path.join(ZIPFORMER_MODEL_DIR, "config.json")

# Khởi tạo Zipformer recognizer
vi_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=os.path.join(ZIPFORMER_MODEL_DIR, "encoder-epoch-20-avg-10.onnx"),
    decoder=os.path.join(ZIPFORMER_MODEL_DIR, "decoder-epoch-20-avg-10.onnx"),
    joiner=os.path.join(ZIPFORMER_MODEL_DIR, "joiner-epoch-20-avg-10.onnx"),
    tokens=tokens_path,
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

print("✅ Zipformer Vietnamese ASR đã sẵn sàng!")

# 4. VITS TTS (Hmong Text -> Speech)
TTS_CONFIG_PATH = "HmongTTS/hmong.json"
TTS_MODEL_PATH = "HmongTTS/G_60000.pth"
device_tts = "cpu"  # TTS thường chạy tốt trên CPU

print(f"🔊 Đang tải VITS TTS model (device: {device_tts})...")
hps = utils.get_hparams_from_file(TTS_CONFIG_PATH)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
).to(device_tts)

net_g.eval()
utils.load_checkpoint(TTS_MODEL_PATH, net_g, None)


def get_text(text, hps):
    """Convert text to sequence for TTS"""
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


print("✅ Tất cả models đã được tải thành công!")

# ==================== RESPONSE MODELS ====================


class TranslationResponse(BaseModel):
    hmong_text: str
    vietnamese_text: str
    success: bool
    message: Optional[str] = None

# ==================== API 1: MÔNG -> VIỆT ====================


@app.post("/api/hmong-to-vietnamese",
          response_model=TranslationResponse,
          summary="Dịch từ tiếng Mông sang tiếng Việt",
          description="Nhận file âm thanh tiếng Mông, chuyển thành text, dịch sang tiếng Việt")
async def hmong_to_vietnamese(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()

        # Lấy extension gốc
        original_ext = os.path.splitext(audio.filename)[1] or ".wav"

        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_input:
            tmp_input.write(audio_bytes)
            tmp_input_path = tmp_input.name

        try:
            # ASR - Nhận dạng giọng nói tiếng Mông
            print(
                f"🎤 Đang nhận dạng giọng nói tiếng Mông từ file {audio.filename}...")
            asr_result = asr_pipeline(tmp_input_path, batch_size=8)
            hmong_text = asr_result["text"].strip()

            if not hmong_text:
                raise HTTPException(
                    status_code=400, detail="Không nhận dạng được văn bản từ audio")

            print(f"📝 Text tiếng Mông: {hmong_text}")

            # Dịch sang tiếng Việt
            print("🌐 Đang dịch sang tiếng Việt...")
            vietnamese_text = translate_with_groq(hmong_text, "hmn", "vi")
            print(f"✅ Text tiếng Việt: {vietnamese_text}")

            return TranslationResponse(
                hmong_text=hmong_text,
                vietnamese_text=vietnamese_text,
                success=True,
                message="Dịch thành công"
            )

        finally:
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


# ==================== API 2: VIỆT -> MÔNG ====================


@app.post("/api/vietnamese-to-hmong",
          summary="Dịch từ tiếng Việt sang tiếng Mông",
          description="Nhận file âm thanh tiếng Việt, chuyển thành text, dịch sang tiếng Mông và tạo audio")
async def vietnamese_to_hmong(audio: UploadFile = File(...)):
    """
    API 2: Dịch từ tiếng Việt sang tiếng Mông với audio output
    """
    try:
        # Đọc file audio
        audio_bytes = await audio.read()

        # Lấy extension gốc
        original_ext = os.path.splitext(audio.filename)[1] or ".wav"

        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_input:
            tmp_input.write(audio_bytes)
            tmp_input_path = tmp_input.name

        try:
            # Bước 1: ASR - Đọc file audio trực tiếp
            print(
                f"🎤 Đang nhận dạng giọng nói tiếng Việt từ file {audio.filename}...")

            audio_data, sample_rate = sf.read(tmp_input_path, dtype='float32')

            # Đảm bảo mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample về 16kHz nếu cần
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            # Tạo stream và nhận dạng
            stream = vi_recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio_data)
            vi_recognizer.decode_stream(stream)
            vietnamese_text = stream.result.text.strip()

            if not vietnamese_text:
                raise HTTPException(
                    status_code=400, detail="Không nhận dạng được văn bản từ audio")

            print(f"📝 Text tiếng Việt: {vietnamese_text}")

            # Bước 2: Dịch sang tiếng Mông
            print("🌐 Đang dịch sang tiếng Mông...")
            hmong_text = translate_with_groq(vietnamese_text, "vi", "hmn")
            print(f"📝 Text tiếng Mông: {hmong_text}")

            # Bước 3: TTS - Tạo audio tiếng Mông
            print("🔊 Đang tạo audio tiếng Mông...")
            stn_tst = get_text(hmong_text, hps)
            with torch.no_grad():
                x_tts = stn_tst.unsqueeze(0).to(device_tts)
                x_tts_lengths = torch.LongTensor(
                    [stn_tst.size(0)]).to(device_tts)
                audio_output = net_g.infer(
                    x_tts,
                    x_tts_lengths,
                    noise_scale=float(os.getenv("NOISE_SCALE", 0.667)),
                    noise_scale_w=float(os.getenv("NOISE_SCALE_W", 0.8)),
                    length_scale=float(os.getenv("LENGTH_SCALE", 1.0))
                )[0][0, 0].data.cpu().float().numpy()

            # Convert to bytes
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_output,
                     hps.data.sampling_rate, format='WAV')
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode()

            return {
                "vietnamese_text": vietnamese_text,
                "hmong_text": hmong_text,
                "audio_base64": audio_base64,
                "audio_format": "wav",
                "sample_rate": hps.data.sampling_rate,
                "success": True,
                "message": "Dịch và tạo audio thành công"
            }

        finally:
            # Xóa file tạm
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


# ==================== API KIỂM TRA ====================
@app.get("/")
async def root():
    """API info endpoint"""
    return {
        "app": "Hmong-Vietnamese Translation API",
        "version": "1.0.0",
        "endpoints": {
            "hmong_to_vietnamese": "/api/hmong-to-vietnamese (POST)",
            "vietnamese_to_hmong": "/api/vietnamese-to-hmong (POST)"
        },
        "status": "ready",
        "models": {
            "asr_hmong": "Whisper (Pakorn2112/whisper-model-large-hmong)",
            "asr_vietnamese": "Zipformer (hynt/Zipformer-30M-RNNT-6000h)",
            "translator": "Google Translate",
            "tts": "VITS (Hmong)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": True}


# ==================== CHẠY SERVER ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
