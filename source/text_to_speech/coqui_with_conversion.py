from typing import List
from TTS.api import TTS
import numpy as np
from .abc import Synthesizer


class CoquiWithVoiceConversionSynthesizer(Synthesizer):

    def __init__(self, presynthesised_phrases: List[str]=None, conversion_wav_path: str='exploratory/voice_conversion3.wav'):
        self.tts = TTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC_ph",
            gpu=False,
        )
        self.sample_rate = self.tts.synthesizer.output_sample_rate
        self.conversion_wav_path = conversion_wav_path

        super().__init__(sample_rate=self.sample_rate, presynthesised_phrases=presynthesised_phrases)
    
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio data.

        Args:
            text (str): The text to synthesize.
        
        Returns:
            np.ndarray: The synthesized audio data.
        """
        return self.tts.tts_with_vc(text, speaker_wav=self.conversion_wav_path)
