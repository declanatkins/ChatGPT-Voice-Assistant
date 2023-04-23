from abc import abstractmethod, ABC
import numpy as np


class Synthesizer(ABC):
    """Abstract base class for synthesizers.
    """

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio data.
        
        Args:
            text (str): The text to synthesize.
        
        Returns:
            np.ndarray: The synthesized audio data.
        """
        pass
