from io import BytesIO
import numpy as np
import requests
from .abc import Synthesizer

class RemoteCoquiSynthesizer(Synthesizer):

    def __init__(self, url: str, sampling_rate: int=16000):
        self.url = url
        self.sampling_rate = sampling_rate
    

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio data.
        
        Args:
            text (str): The text to synthesize.
        
        Returns:
            np.ndarray: The synthesized audio data.
        """
        response = requests.post(self.url, data=text.encode("utf-8"))
        return np.load(BytesIO(response.content))