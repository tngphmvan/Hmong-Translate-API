import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# Configure paths
repo_id = "Pakorn2112/whisper-model-large-hmong"
subfolder_name = "SingleSpeech"
audio_file = "/kaggle/input/hmong-sample/download (1).wav"

# Check GPU 
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng thiết bị: {device}")

# 1. Manually load the model and processor from the subfolder
# This step helps avoid errors where weight files (bin/safetensors) are not found
print("Đang tải model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    repo_id,
    subfolder=subfolder_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(repo_id, subfolder=subfolder_name)

# 2. Initialize the pipeline using the preloaded model
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Important for Whisper to handle long audio files
    device=device,
)

# 3. Run speech recognition
print("Đang xử lý file âm thanh...")
# batch_size=8 helps speed up inference if a GPU is available
result = transcriber(audio_file, batch_size=8)

print("--- KẾT QUẢ ---")
print(result["text"])
