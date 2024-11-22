import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import logging
from typing import Union, Optional


class Transcriber:

    def __init__(self):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3-turbo"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    def load_audio(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load and preprocess audio file using librosa.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio array or None if loading fails
        """
        try:
            # Convert to Path object for better path handling
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # Load audio file
            self.logger.info(f"Loading audio file: {audio_path}")
            audio_array, sampling_rate = librosa.load(
                audio_path,
                sr=16000,  # Whisper expects 16kHz sampling rate
                mono=True
            )
            
            # Normalize audio
            audio_array = librosa.util.normalize(audio_array)
            
            return audio_array
            
        except Exception as e:
            self.logger.error(f"Error loading audio file: {str(e)}")
            raise

    def transcribe(self, audio_file: Union[str, Path]) -> dict:
        """
        Transcribe audio file using the Whisper model.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Dictionary containing transcription results and metadata
        """
        try:
            # Load and preprocess audio
            audio_array = self.load_audio(audio_file)
            
            if audio_array is None:
                raise ValueError("Failed to load audio file")

            # Get audio duration
            duration = librosa.get_duration(y=audio_array, sr=16000)
            
            # Process with pipeline
            self.logger.info("Starting transcription...")
            result = self.pipe(
                audio_array,
                chunk_length_s=30,  # Process in 30-second chunks
                batch_size=4,       # Adjust based on your GPU memory
                return_timestamps=True
            )
            
            # Add metadata to results
            result.update({
                "duration": duration,
                "sample_rate": 16000,
                "status": "success"
            })
            
            self.logger.info("Transcription completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
                "text": None
            }







transcriber = Transcriber()
transcription = transcriber.transcribe("downloads/xgs5gOCpsAE.wav")
print("Transcription: ", transcription)