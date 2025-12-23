"""
Kaggle-Optimized Hmong TTS Batch Processing Script

This script is optimized for running in Kaggle notebooks with proper
dependency installation and GPU utilization if available.

To use in Kaggle:
1. Upload your Excel file to Kaggle input data
2. Run the installation cell first
3. Run the main processing cell

Installation cell:
!pip install -q TTS pandas openpyxl torch torchaudio librosa soundfile scipy
"""

# ============================================================================
# INSTALLATION CELL - Run this first in Kaggle
# ============================================================================
def install_dependencies():
    """Install all required dependencies for Kaggle environment."""
    print("Installing dependencies...")
    import subprocess
    import sys
    
    packages = [
        'pandas',
        'openpyxl',
        'TTS',
        'torch',
        'torchaudio',
        'librosa',
        'soundfile',
        'scipy',
        'numpy'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    
    print("All dependencies installed successfully!")


# ============================================================================
# MAIN PROCESSING CELL
# ============================================================================
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

# Configure logging for Kaggle
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class KaggleHmongTTS:
    """Kaggle-optimized Hmong TTS processor."""
    
    def __init__(self, output_dir: str = '/kaggle/working/audio_output', tts_suffix: str = "_tts"):
        """Initialize the TTS processor for Kaggle environment."""
        self.output_dir = Path(output_dir)
        self.tts_suffix = tts_suffix
        self.tts_model = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Check for GPU availability
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
        except ImportError:
            self.device = "cpu"
            logger.info("PyTorch not available, using CPU")
    
    def initialize_tts_model(self):
        """Initialize TTS model with Kaggle optimizations."""
        try:
            from TTS.api import TTS
            
            logger.info("Loading TTS model...")
            
            # Use a fast, reliable model for multilingual support
            # Try different models in order of preference
            models_to_try = [
                "tts_models/multilingual/multi-dataset/your_tts",
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/fast_pitch",
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting to load: {model_name}")
                    self.tts_model = TTS(model_name=model_name, progress_bar=True, gpu=(self.device=="cuda"))
                    logger.info(f"Successfully loaded: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load {model_name}: {e}")
                    continue
            
            if self.tts_model is None:
                raise RuntimeError("Failed to load any TTS model")
                
        except ImportError as e:
            logger.error(f"TTS library not found: {e}")
            logger.error("Run: !pip install TTS")
            raise
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            raise
    
    def process_excel(self, excel_path: str):
        """
        Process Excel file with Hmong transcripts.
        
        Args:
            excel_path: Path to Excel file (can be in /kaggle/input/)
        """
        # Read Excel file
        logger.info(f"Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # Validate columns
        if 'file_name' not in df.columns or 'transcript' not in df.columns:
            raise ValueError("Excel must have 'file_name' and 'transcript' columns")
        
        logger.info(f"Loaded {len(df)} rows")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Initialize TTS
        self.initialize_tts_model()
        
        # Process each row
        success_count = 0
        failed_count = 0
        
        for idx, row in df.iterrows():
            try:
                file_name = str(row['file_name']).strip()
                transcript = str(row['transcript']).strip()
                
                if not transcript or transcript.lower() == 'nan':
                    logger.warning(f"Skipping empty transcript for: {file_name}")
                    continue
                
                # Create output filename
                base_name = os.path.splitext(file_name)[0]
                output_filename = f"{base_name}{self.tts_suffix}.wav"
                output_path = self.output_dir / output_filename
                
                logger.info(f"Processing [{idx+1}/{len(df)}]: {file_name}")
                logger.info(f"  Text: {transcript[:80]}...")
                
                # Generate speech
                self.tts_model.tts_to_file(
                    text=transcript,
                    file_path=str(output_path)
                )
                
                logger.info(f"  ✓ Saved: {output_path.name}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                failed_count += 1
                continue
        
        # Print summary
        self.print_summary(len(df), success_count, failed_count)
    
    def print_summary(self, total, success, failed):
        """Print processing summary."""
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        print(f"Total entries:          {total}")
        print(f"Successfully processed: {success}")
        print(f"Failed:                 {failed}")
        if total > 0:
            print(f"Success rate:           {success/total*100:.1f}%")
        else:
            print(f"Success rate:           N/A (no entries)")
        print(f"Output directory:       {self.output_dir}")
        print("="*70)
        
        # List generated files
        wav_files = list(self.output_dir.glob("*.wav"))
        print(f"\nGenerated {len(wav_files)} WAV files:")
        for wav_file in sorted(wav_files)[:10]:  # Show first 10
            print(f"  - {wav_file.name}")
        if len(wav_files) > 10:
            print(f"  ... and {len(wav_files) - 10} more files")


# ============================================================================
# USAGE EXAMPLE FOR KAGGLE NOTEBOOK
# ============================================================================
def kaggle_example_usage():
    """
    Example usage in Kaggle notebook.
    
    Uncomment and modify paths as needed.
    """
    # Initialize processor
    processor = KaggleHmongTTS(
        output_dir='/kaggle/working/hmong_audio_output',
        tts_suffix='_tts'
    )
    
    # Process your Excel file
    # Adjust the path to your input file in Kaggle
    excel_path = '/kaggle/input/your-dataset/hmong_transcripts.xlsx'
    
    processor.process_excel(excel_path)
    
    print("\nDone! Download the audio files from the output directory.")


# ============================================================================
# QUICK START CODE FOR KAGGLE
# ============================================================================
def quick_start(excel_path: str, output_dir: str = '/kaggle/working/audio_output'):
    """
    Quick start function for Kaggle notebooks.
    
    Args:
        excel_path: Path to your Excel file
        output_dir: Where to save audio files (default: /kaggle/working/audio_output)
    
    Example:
        quick_start('/kaggle/input/my-dataset/transcripts.xlsx')
    """
    processor = KaggleHmongTTS(output_dir=output_dir)
    processor.process_excel(excel_path)


# ============================================================================
# RUN THIS IN KAGGLE
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Hmong TTS Batch Processing for Kaggle")
    print("="*70)
    print("\nTo use this script in Kaggle:")
    print("1. First run: install_dependencies()")
    print("2. Then run: quick_start('/kaggle/input/your-dataset/your-file.xlsx')")
    print("\nOr use the KaggleHmongTTS class for more control.")
    print("="*70)
