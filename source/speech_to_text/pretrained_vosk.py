import json
from io import BytesIO
from scipy.io.wavfile import write
from vosk import Model, KaldiRecognizer
import numpy as np
from .abc import Transcriber


class PretrainedVoskTranscriber(Transcriber):

    def __init__(self, model_path="", sampling_rate: int=16000):
        super().__init__(sampling_rate=sampling_rate)

        if model_path:
            self.model = Model(model_path)
        else:
            self.model = Model(lang="en-us")
        self.rec = KaldiRecognizer(self.model, sampling_rate)
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data (np.ndarray): The audio data to transcribe.
        
        Returns:
            str: The transcribed text.
        """

        # Convert audio data to bytes

        buffer = BytesIO()
        write(buffer, self.sampling_rate, audio_data)
        audio_bytes = buffer.read()

        self.rec.AcceptWaveform(audio_bytes)
        return json.loads(self.rec.Result())['text']
