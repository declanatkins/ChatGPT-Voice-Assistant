import os
import numpy as np
from sklearn.mixture import GaussianMixture
from .abc import VAD
from .features import extract_features


class GMMVAD(VAD):
    """Voice Activity Detection using Gaussian Mixture Models
    """

    def __init__(
            self,
            model_path: str='exploratory/gmm_voice_activity_detection',
            window_size: int=4000,
        ):
        """Initializes GMMVAD

        Args:
            model_path (str): Path to GMM model
            window_size (int, optional): Window size in frames, should match trained
                window size. Defaults to 4000.
        """
        self.gmm = GaussianMixture(n_components=2)
        self.gmm.weights_ = np.load(os.path.join(model_path, 'weights.npy'))
        self.gmm.means_ = np.load(os.path.join(model_path, 'means.npy'))
        self.gmm.covariances_ = np.load(os.path.join(model_path, 'covariances.npy'))
        self.gmm_speech_cluster = int(np.load(os.path.join(model_path, 'speech_cluster.npy')))
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
            prediction = self.gmm.predict(features)
            voice_activity.append(int(prediction == self.gmm_speech_cluster))
        
        return np.array(voice_activity)
