import librosa
import numpy as np
from scipy.signal import find_peaks


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extracts features from audio signal

    The features extracted are:
    - MFCCs
    - Zero crossing rate (ZCR)
    - Short-term energy (STE)
    - Pitch

    Args:
        y (np.ndarray): Audio signal
        sr (int): Sampling rate of audio signal
    Returns:
        np.ndarray: Extracted feature vector
    """

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Extract zero crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)

    # Extract short-term energy (STE)
    ste = librosa.feature.rms(y)
    
    # Extract pitch using autocorrelation
    autocorr = np.correlate(y, y, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks, _ = find_peaks(autocorr)
    pitch = sr / peaks[0] if peaks.size > 0 else np.array([0])
    # Concatenate features
    features = np.concatenate([
        mfcc.flatten(),
        zcr.flatten(),
        ste.flatten(),
        pitch.flatten(),
    ])
    return features
