#!/bin/bash
# Installation script for Kaggle environment
# Run this in Kaggle notebook: !bash install_kaggle.sh

echo "=================================================="
echo "Installing Hmong TTS Dependencies for Kaggle"
echo "=================================================="

pip install -q pandas openpyxl TTS torch torchaudio librosa soundfile scipy numpy

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "You can now use the Hmong TTS scripts:"
echo "  from kaggle_hmong_tts import quick_start"
echo "  quick_start('/kaggle/input/your-dataset/your-file.xlsx')"
echo ""
