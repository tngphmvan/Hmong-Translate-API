import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# Cấu hình đường dẫn
repo_id = "Pakorn2112/whisper-model-large-hmong"
subfolder_name = "SingleSpeech"
audio_file = "/kaggle/input/hmong-sample/download (1).wav"

# Kiểm tra GPU (Chạy Whisper Large trên CPU rất chậm, nên dùng GPU)
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng thiết bị: {device}")

# 1. Tải Model và Processor thủ công từ subfolder
# Bước này giúp tránh lỗi không tìm thấy file trọng số (bin/safetensors)
print("Đang tải model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    repo_id,
    subfolder=subfolder_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(repo_id, subfolder=subfolder_name)

# 2. Khởi tạo pipeline với object model đã tải
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Quan trọng cho Whisper để xử lý file dài
    device=device,
)

# 3. Chạy nhận dạng
print("Đang xử lý file âm thanh...")
# batch_size=8 giúp chạy nhanh hơn nếu có GPU
result = transcriber(audio_file, batch_size=8)

print("--- KẾT QUẢ ---")
print(result["text"])
