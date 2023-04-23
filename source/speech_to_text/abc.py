from abc import abstractmethod, ABC
import numpy as np


class Transcriber(ABC):
    """Abstract base class for transcribers.
    """

    def __init__(self, sampling_rate) -> None:
        self.sampling_rate = sampling_rate

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text.
        
        Args:
            audio_data (np.ndarray): The audio data to transcribe.
        
        Returns:
            str: The transcribed text.
        """
        pass
