from io import BytesIO
import requests
from .abc import AudioDataType, Transcriber



class RemoteVoskTranscriber(Transcriber):
    """ Remote Vosk Transcriber
    """

    def __init__(self, url: str, sampling_rate: int=16000):
        super().__init__(sampling_rate=sampling_rate)
        self.url = url

    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data (bytes): The audio data to transcribe.
        
        Returns:
            str: The transcribed text.
        """
        response = requests.post(self.url, files={'audio': BytesIO(audio_data)})
        return response.json()['text']

    @property
    def audio_data_type(self) -> AudioDataType:
        return AudioDataType.BYTES