import numpy as np
import whisper
from .abc import Transcriber


class PretrainedWhisperTranscriber(Transcriber):
    """ Pretrained Whisper Transcriber
    """

    def __init__(self, model_path="", sampling_rate: int=16000):
        super().__init__(sampling_rate=sampling_rate)

        if model_path:
            self.model = whisper.load_model(model_path)
        else:
            self.model = whisper.load_model('base')
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data (np.ndarray): The audio data to transcribe.
        
        Returns:
            str: The transcribed text.
        """

        return self.model.transcribe(audio_data)['text']
