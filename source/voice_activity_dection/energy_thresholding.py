from .abc import VAD
import numpy as np
from scipy.signal import filtfilt, butter


def bandpass_filter(signal_data, lowcut, highcut, sample_rate):
    """
    Apply band-pass filter to signal data.
    
    Args:
        signal_data (np.ndarray): Input audio signal data.
        lowcut (float): Lower cutoff frequency for band-pass filter.
        highcut (float): Upper cutoff frequency for band-pass filter.
        sample_rate (int): Sampling frequency of the signal data.
        
    Returns:
        filtered_data (np.ndarray): Filtered signal data.
    """
    # Normalize cutoff frequencies
    low = lowcut / (sample_rate / 2)
    high = highcut / (sample_rate / 2)

    # Design Butterworth band-pass filter
    b, a = butter(1, [low, high], btype='band')

    # Apply filter to signal data
    filtered_data = filtfilt(b, a, signal_data)

    return filtered_data


class EnergyThresholdingVAD(VAD):
    """Voice Activity Detection using energy thresholding
    """

    def __init__(
            self,
            threshold: float=0.7,
            overlap_ratio: float=0.8,
            window_size: int=2000,
        ):
        """Initializes EnergyThresholdingVAD

        Args:
            threshold (float, optional): Threshold for energy. Defaults to 0.7.
            overlap_ratio (float, optional): Overlap ratio for window. Defaults to 0.8.
            window_size (int, optional): Window size in frames. Defaults to 2000.
        """
        self.threshold = threshold
        self.overlap_ratio = overlap_ratio
        self.window_size = window_size
    
    def detect(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Detects voice activity in audio

        Args:
            audio (np.ndarray): Audio signal

        Returns:
            np.ndarray: Voice activity mask
        """

            # Define human speech frequency range
        lowcut = 100
        highcut = 3200

        # Apply band-pass filter to signal data
        filtered_data = bandpass_filter(audio, lowcut, highcut, sample_rate)

        # Calculate energy of filtered signal data
        window = np.hamming(self.window_size)
        hop_length = int(self.window_size * (1 - self.overlap_ratio))
        energy = np.array([
            np.sum(window * filtered_data[i:i + self.window_size] ** 2)
            for i in range(0, len(filtered_data) - self.window_size, hop_length)
        ])

        # Calculate total energy
        total_energy = np.array([
            np.sum(window * audio[i:i + self.window_size] ** 2)
            for i in range(0, len(audio) - self.window_size, hop_length)
        ])

        energy_ratio = energy / total_energy
        # print(energy_ratio)
        windowed_voice_activity = []

        # Compare energy ratio with threshold
        for i in range(len(energy_ratio)):
            if energy_ratio[i] > self.threshold:
                windowed_voice_activity.append(1)  # Speech
            else:
                windowed_voice_activity.append(0)  # Silent pause
        
        # Convert from overlapping windows to signal length predictions
        voice_activity = np.zeros(len(audio))
        for i in range(len(windowed_voice_activity)):
            voice_activity[i * hop_length:(i + 1) * hop_length] = windowed_voice_activity[i]
        return voice_activity