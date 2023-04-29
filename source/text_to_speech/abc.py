from abc import abstractmethod, ABC
from typing import List
import numpy as np
import sounddevice as sd

class Synthesizer(ABC):
    """Abstract base class for synthesizers.
    """

    def __init__(self, sample_rate: int=16000, presynthesised_phrases: List[str]=None):
        """Initializes Synthesizer
        
        Args:
            sample_rate (int, optional): Sample rate. Defaults to 16000.
        """
        self.sample_rate = sample_rate
        if presynthesised_phrases is None:
            presynthesised_phrases = []
        
        self.presynthesised_phrases = {}
        print('Presynthesising phrases...')
        for phrase in presynthesised_phrases:
            self.presynthesised_phrases[phrase] = self.synthesize(phrase)
        print('Finished presynthesising phrases.')

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

        if text in self.presynthesised_phrases:
            audio = self.presynthesised_phrases[text]
        else:
            audio = self.synthesize(text)
        if isinstance(audio, list):
            audio = np.array(audio)
        sd.play(audio, self.sample_rate, blocking=True)
        