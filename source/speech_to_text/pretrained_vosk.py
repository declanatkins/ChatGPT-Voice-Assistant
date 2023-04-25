import json

from vosk import Model, KaldiRecognizer
from .abc import AudioDataType, Transcriber


class PretrainedVoskTranscriber(Transcriber):

    def __init__(self, model_path="", sampling_rate: int=16000):
        super().__init__(sampling_rate=sampling_rate)

        if model_path:
            self.model = Model(model_path)
        else:
            self.model = Model(lang="en-us")
        self.rec = KaldiRecognizer(self.model, sampling_rate)
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data (bytes): The audio data to transcribe.
        
        Returns:
            str: The transcribed text.
        """
        self.rec.AcceptWaveform(audio_data)
        return json.loads(self.rec.Result())['text']

    @property
    def audio_data_type(self) -> AudioDataType:
        return AudioDataType.BYTES
