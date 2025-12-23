import os
import subprocess

"""
Script chạy Text-to-Speech (TTS) cho tiếng H'Mông dựa trên mô hình VITS.

Chức năng chính:
- Tự động compile module C++ alignment (phục vụ inference)
- Load mô hình TTS đã train
- Cung cấp giao diện web thông qua Gradio
- Sinh audio từ văn bản tiếng H'Mông

Script này phù hợp để deploy demo trên Hugging Face Spaces hoặc môi trường cloud.
"""

# -------------------------------------------------------------------------
# 1. AUTO-COMPILE FOR LINUX (Hugging Face)
# -------------------------------------------------------------------------
"""
Do mô hình được train trên macOS, khi chạy trên Linux (Hugging Face / cloud),
cần build lại module C++ `monotonic_align` để đảm bảo tương thích hệ điều hành.
"""
print("⚙️ Compiling Alignment Tool...")
cwd = os.getcwd()
os.chdir("monotonic_align")
subprocess.run(["python", "setup.py", "build_ext", "--inplace"])
os.chdir(cwd)
print("✅ Compilation Complete.")

import sys
# Đảm bảo Python có thể tìm thấy module C++ vừa compile
sys.path.append(os.getcwd())

import torch
import commons
import utils
import json
import gradio as gr
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

# -------------------------------------------------------------------------
# 2. CONFIGURATION
# -------------------------------------------------------------------------
"""
Các file cấu hình và checkpoint của mô hình TTS.
"""
CONFIG_PATH = "hmong.json"      # File config của mô hình
MODEL_PATH = "G_60000.pth"      # Checkpoint mô hình (đổi nếu tên khác)
LEXICON_PATH = "lexicon.json"   # Lexicon (không dùng trực tiếp trong script)

# -------------------------------------------------------------------------
# 3. LOAD HELPERS & MODEL
# -------------------------------------------------------------------------

"""
Load mapping speaker name -> speaker ID.
Mặc định chỉ có 1 speaker nếu không tồn tại file speakers.json.
"""
speakers = {"Default": 0}
if os.path.exists("speakers.json"):
    with open("speakers.json", "r") as f:
        speakers = json.load(f)

# Danh sách tên speaker dùng cho dropdown UI
speaker_names = list(speakers.keys())

"""
Load hyperparameters từ file config.
"""
hps = utils.get_hparams_from_file(CONFIG_PATH)

"""
Thiết bị chạy inference.
Cloud demo thường dùng CPU và vẫn đủ nhanh.
"""
device = "cpu"

"""
Khởi tạo mô hình VITS (SynthesizerTrn).
"""
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
).to(device)

# Chuyển mô hình sang chế độ inference
_ = net_g.eval()

"""
Load trọng số mô hình từ checkpoint.
"""
utils.load_checkpoint(MODEL_PATH, net_g, None)

# -------------------------------------------------------------------------
# 4. TEXT PREPROCESSING
# -------------------------------------------------------------------------
def get_text(text, hps):
    """
    Chuyển văn bản đầu vào thành tensor ID ký tự dùng cho mô hình.

    Các bước:
    - Text cleaning
    - Convert sang sequence ký tự
    - Thêm blank token nếu cần
    - Convert sang torch.LongTensor

    :param text: Văn bản tiếng H'Mông
    :param hps: Hyperparameters của mô hình
    :return: Tensor LongTensor biểu diễn chuỗi văn bản
    """
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# -------------------------------------------------------------------------
# 5. TTS INFERENCE FUNCTION (GRADIO)
# -------------------------------------------------------------------------
def tts_fn(text, speaker_name):
    """
    Hàm sinh audio từ văn bản tiếng H'Mông.
    Được Gradio gọi trực tiếp khi người dùng nhấn submit.

    :param text: Văn bản tiếng H'Mông đầu vào
    :param speaker_name: Tên speaker được chọn
    :return: Tuple (sampling_rate, audio_numpy_array)
    """
    if not text:
        return None

    # Loại bỏ các ký tự không nằm trong tập symbols
    text = "".join([c for c in text if c in symbols])

    # Lấy speaker ID
    sid = speakers.get(speaker_name, 0)

    # Xử lý text thành tensor
    stn_tst = get_text(text, hps)

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid_tensor = torch.LongTensor([sid]).to(device)

        # Inference mô hình VITS
        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            sid=sid_tensor,
            noise_scale=.667,
            noise_scale_w=0.8,
            length_scale=1
        )[0][0, 0].data.cpu().float().numpy()

    return (hps.data.sampling_rate, audio)

# -------------------------------------------------------------------------
# 6. LAUNCH GRADIO WEB APP
# -------------------------------------------------------------------------
"""
Khởi tạo giao diện web Gradio cho demo TTS.
"""
app = gr.Interface(
    fn=tts_fn,
    inputs=[
        gr.Textbox(label="Enter Hmong Text", placeholder="Nyob zoo..."),
        gr.Dropdown(
            choices=speaker_names,
            value=speaker_names[0],
            label="Speaker"
        )
    ],
    outputs=gr.Audio(label="Output"),
    title="Hmong TTS Demo",
    description="A VITS-based Text-to-Speech model for Hmong Daw and Hmong Njua."
)

# Chạy web app
app.launch()
