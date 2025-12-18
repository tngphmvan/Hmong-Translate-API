import os
import subprocess

# 1. AUTO-COMPILE FOR LINUX (Hugging Face)
# Since you trained on Mac, the cloud needs to rebuild the C++ aligner.
print("⚙️ Compiling Alignment Tool...")
cwd = os.getcwd()
os.chdir("monotonic_align")
subprocess.run(["python", "setup.py", "build_ext", "--inplace"])
os.chdir(cwd)
print("✅ Compilation Complete.")

import sys
sys.path.append(os.getcwd()) # Ensure python finds the compiled module

import torch
import commons
import utils
import json
import gradio as gr
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

# 2. CONFIGURATION
CONFIG_PATH = "hmong.json" # We moved this to root
MODEL_PATH = "G_60000.pth" # CHANGE THIS to your exact filename
LEXICON_PATH = "lexicon.json"

# 3. LOAD HELPERS
# Load Speaker Map
speakers = {"Default": 0}
if os.path.exists("speakers.json"):
    with open("speakers.json", "r") as f:
        speakers = json.load(f)
# Reverse map for dropdown (Name -> ID)
speaker_names = list(speakers.keys())

# Load Model
hps = utils.get_hparams_from_file(CONFIG_PATH)
device = "cpu" # Cloud demos usually run on CPU (it's fast enough for inference)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)

_ = net_g.eval()
utils.load_checkpoint(MODEL_PATH, net_g, None)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# 4. THE WEB FUNCTION
def tts_fn(text, speaker_name):
    if not text: return None
    
    # Clean invalid chars manually just in case
    text = "".join([c for c in text if c in symbols])
    
    sid = speakers.get(speaker_name, 0)
    stn_tst = get_text(text, hps)
    
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid_tensor = torch.LongTensor([sid]).to(device)
        
        audio = net_g.infer(
            x_tst, 
            x_tst_lengths, 
            sid=sid_tensor, 
            noise_scale=.667, 
            noise_scale_w=0.8, 
            length_scale=1
        )[0][0,0].data.cpu().float().numpy()
        
    return (hps.data.sampling_rate, audio)

# 5. LAUNCH WEBSITE
app = gr.Interface(
    fn=tts_fn,
    inputs=[
        gr.Textbox(label="Enter Hmong Text", placeholder="Nyob zoo..."),
        gr.Dropdown(choices=speaker_names, value=speaker_names[0], label="Speaker")
    ],
    outputs=gr.Audio(label="Output"),
    title="Hmong TTS Demo",
    description="A VITS-based Text-to-Speech model for Hmong Daw and Hmong Njua."
)

app.launch()