import numpy as np
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from .abc import VAD
from .features import extract_features


class XGBVAD(VAD):
    """Voice Activity Detection using XGBoost
    """

    def __init__(
            self,
            model_path: str='exploratory/xgb_voice_activity_detection.json',
            window_size: int=4000,
        ):
        """Initializes XGBVAD

        Args:
            model_path (str): Path to XGBoost model
            window_size (int, optional): Window size in frames, should match trained
                window size. Defaults to 4000.
        """
        self.xgb = XGBClassifier()
        self.xgb.load_model(model_path)
        self.window_size = window_size

    def detect(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Detects voice activity in audio

        Args:
            audio (np.ndarray): Audio signal

        Returns:
            np.ndarray: Voice activity mask
        """
        
        # Split audio into windows
        windows = []
        for i in range(0, len(audio), self.window_size):
            windows.append(audio[i:i + self.window_size])
        
        features = []
        for window in windows:
            features.append(extract_features(window, sample_rate))
        voice_activity = self.xgb.predict(np.array(features))
    
        # Covert from windows to signal length predictions
        voice_activity_signal = np.zeros(len(audio))
        for i in range(len(windows)):
            voice_activity_signal[i * self.window_size:(i + 1) * self.window_size] = voice_activity[i]
        
        return voice_activity_signal
