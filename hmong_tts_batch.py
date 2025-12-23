#!/usr/bin/env python3
"""
Hmong Text-to-Speech Batch Processing Script

This script processes an Excel file with transcripts and generates speech audio files
for Hmong language text using Text-to-Speech technology.

Usage:
    python hmong_tts_batch.py --input input.xlsx --output ./output_folder

Requirements:
    - pandas
    - openpyxl (for Excel file reading)
    - TTS (Coqui TTS) or alternative TTS library
    - torch (for deep learning models)
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HmongTTSProcessor:
    """Process Hmong text to speech conversion in batch mode."""
    
    def __init__(self, output_dir: str, tts_suffix: str = "_tts"):
        """
        Initialize the TTS processor.
        
        Args:
            output_dir: Directory to save generated WAV files
            tts_suffix: Suffix to add to output filenames
        """
        self.output_dir = Path(output_dir)
        self.tts_suffix = tts_suffix
        self.tts_model = None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
    def initialize_tts_model(self, model_name: str = "tts_models/multilingual/multi-dataset/your_tts"):
        """
        Initialize the TTS model.
        
        Args:
            model_name: Name of the TTS model to use
        """
        try:
            from TTS.api import TTS
            logger.info(f"Loading TTS model: {model_name}")
            
            # Try to use multilingual model that might support Hmong
            # If specific Hmong model not available, use closest language model
            try:
                self.tts_model = TTS(model_name=model_name)
                logger.info("TTS model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
                # Fallback to a basic multilingual model
                logger.info("Attempting to load fallback model...")
                self.tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                logger.info("Fallback TTS model loaded")
                
        except ImportError:
            logger.error("TTS library not found. Please install: pip install TTS")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error initializing TTS model: {e}")
            sys.exit(1)
    
    def read_excel_file(self, excel_path: str) -> pd.DataFrame:
        """
        Read the Excel file with file_name and transcript columns.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            DataFrame with the Excel data
        """
        try:
            logger.info(f"Reading Excel file: {excel_path}")
            df = pd.read_excel(excel_path)
            
            # Validate required columns
            required_columns = ['file_name', 'transcript']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Loaded {len(df)} rows from Excel file")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Excel file not found: {excel_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            sys.exit(1)
    
    def generate_speech(self, text: str, output_path: str) -> bool:
        """
        Generate speech audio from text and save to file.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not text or pd.isna(text):
                logger.warning(f"Empty text, skipping: {output_path}")
                return False
            
            text = str(text).strip()
            if not text:
                logger.warning(f"Empty text after stripping, skipping: {output_path}")
                return False
            
            logger.info(f"Generating speech for: {output_path}")
            logger.debug(f"Text: {text[:100]}...")
            
            # Generate speech using TTS model
            self.tts_model.tts_to_file(
                text=text,
                file_path=output_path
            )
            
            logger.info(f"Successfully generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating speech for {output_path}: {e}")
            return False
    
    def process_batch(self, excel_path: str) -> dict:
        """
        Process all entries in the Excel file.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Dictionary with processing statistics
        """
        # Read Excel file
        df = self.read_excel_file(excel_path)
        
        # Initialize TTS model
        self.initialize_tts_model()
        
        # Statistics
        stats = {
            'total': len(df),
            'success': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Process each row
        for idx, row in df.iterrows():
            file_name = row.get('file_name', f'audio_{idx}')
            transcript = row.get('transcript', '')
            
            # Clean filename and add TTS suffix
            if pd.isna(file_name):
                file_name = f'audio_{idx}'
            
            file_name = str(file_name).strip()
            
            # Remove extension if present and add TTS suffix
            base_name = os.path.splitext(file_name)[0]
            output_filename = f"{base_name}{self.tts_suffix}.wav"
            output_path = self.output_dir / output_filename
            
            # Generate speech
            logger.info(f"Processing [{idx + 1}/{stats['total']}]: {file_name}")
            
            if self.generate_speech(transcript, str(output_path)):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        return stats
    
    def print_summary(self, stats: dict):
        """Print processing summary."""
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total entries: {stats['total']}")
        logger.info(f"Successfully processed: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Batch process Hmong text to speech from Excel file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python hmong_tts_batch.py --input data.xlsx --output ./audio_output
    python hmong_tts_batch.py --input data.xlsx --output ./audio_output --suffix _hmong_tts
    
Excel file format:
    Required columns:
        - file_name: Name for the output audio file (without extension)
        - transcript: Hmong text to convert to speech
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input Excel file (.xlsx)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output directory for WAV files'
    )
    
    parser.add_argument(
        '--suffix', '-s',
        default='_tts',
        help='Suffix to add to output filenames (default: _tts)'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='tts_models/multilingual/multi-dataset/your_tts',
        help='TTS model to use (default: multilingual model)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create processor and run
    processor = HmongTTSProcessor(
        output_dir=args.output,
        tts_suffix=args.suffix
    )
    
    try:
        stats = processor.process_batch(args.input)
        processor.print_summary(stats)
        
        if stats['failed'] > 0:
            logger.warning(f"{stats['failed']} items failed to process")
            sys.exit(1)
        else:
            logger.info("All items processed successfully!")
            
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
