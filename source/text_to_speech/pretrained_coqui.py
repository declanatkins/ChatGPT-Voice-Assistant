from TTS.api import TTS
import numpy as np
from .abc import Synthesizer


class PretrainedCoquiSynthesizer(Synthesizer):

    def __init__(self):
        self.tts = TTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC_ph",
            gpu=False,
        )
        self.sample_rate = self.tts.synthesizer.output_sample_rate
    
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio data.

        Args:
            text (str): The text to synthesize.
        
        Returns:
            np.ndarray: The synthesized audio data.
        """
        return self.tts.tts(text)
