import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from .abc import AudioDataType, Transcriber


class CustomWhisperTranscriber(Transcriber):
    """ Pretrained Whisper Transcriber
    """

    def __init__(self, model_path, sampling_rate: int=16000):
        super().__init__(sampling_rate=sampling_rate)

        self.processor = WhisperProcessor.from_pretrained('openai/whisper-small')
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data (np.ndarray): The audio data to transcribe.
        
        Returns:
            str: The transcribed text.
        """

        input_features = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)[0]
        transcription = self.processor.decode(predicted_ids)
        return self.processor.tokenizer._normalize(transcription)

    @property
    def audio_data_type(self) -> AudioDataType:
        return AudioDataType.NUMPY
