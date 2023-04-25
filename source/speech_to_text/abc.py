from abc import abstractmethod, ABC
from enum import Enum
from typing import Union
import numpy as np


class AudioDataType(Enum):
    """The type of audio data that a transcriber can handle.
    """
    BYTES = "bytes"
    NUMPY = "numpy"


class Transcriber(ABC):
    """Abstract base class for transcribers.
    """

    def __init__(self, sampling_rate) -> None:
        self.sampling_rate = sampling_rate

    @abstractmethod
    def transcribe(self, audio_data: Union[np.ndarray, bytes]) -> str:
        """Transcribe audio data to text.
        
        Args:
            audio_data (Union[np.ndarray, bytes]): The audio data to transcribe. The type is determined by
                the method get_audio_data_type().
        
        Returns:
            str: The transcribed text.
        """
        raise NotImplementedError("Cannot call abstract method transcribe()")
    
    @property
    @abstractmethod
    def audio_data_type(self) -> AudioDataType:
        raise NotImplementedError("Cannot call abstract method get_audio_data_type()")
    
