from abc import ABC, abstractmethod
import numpy as np


class VAD(ABC):
    """Abstract class for voice activity detection
    """

    @abstractmethod
    def detect(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Detects voice activity in audio

        Args:
            audio (np.ndarray): Audio signal

        Returns:
            np.ndarray: Voice activity mask
        """
        raise NotImplementedError('detect() must be implemented in subclass')
