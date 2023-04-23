import numpy as np
from xgb import XGBClassifier
from .abc import VAD
from .features import extract_features


class XGBVAD(VAD):
    """Voice Activity Detection using XGBoost
    """

    def __init__(
            self,
            model_path: str='exploratory/xgb_voice_activity_detection',
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
            np.ndarray: Voice activity for each window
        """
        
        # Split audio into windows
        windows = []
        for i in range(0, len(audio), self.window_size):
            windows.append(audio[i:i + self.window_size])
        
        # Extract features from windows and predict
        voice_activity = []
        for window in windows:
            features = extract_features(window, sample_rate)
            prediction = self.xgb.predict(features)
            voice_activity.append(int(prediction))
        
        return np.array(voice_activity)