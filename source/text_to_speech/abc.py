from abc import abstractmethod, ABC
import numpy as np
import sounddevice as sd

class Synthesizer(ABC):
    """Abstract base class for synthesizers.
    """

    def __init__(self, sample_rate: int=16000):
        """Initializes Synthesizer
        
        Args:
            sample_rate (int, optional): Sample rate. Defaults to 16000.
        """
        self.sample_rate = sample_rate

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio data.
        
        Args:
            text (str): The text to synthesize.
        
        Returns:
            np.ndarray: The synthesized audio data.
        """
        pass
    
    def say(self, text: str):
        """Synthesize text to audio data and play it.
        
        Args:
            text (str): The text to synthesize.
        """
        audio = self.synthesize(text)
        if isinstance(audio, list):
            audio = np.array(audio)
        sd.play(audio, self.sample_rate, blocking=True)
        