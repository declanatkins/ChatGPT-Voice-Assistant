from typing import Generator, List
import numpy as np
import sounddevice
import threading
from .voice_activity_dection import VAD


def get_consecutive_sequences(arr: np.ndarray) -> List[np.ndarray]:
    """
    Convert a numpy array containing only 0s and 1s into a list of arrays
    of sequences of consecutive equal values in order.
    
    Args:
    arr (numpy.ndarray): Input numpy array containing only 0s and 1s.
    
    Returns:
    list: List of numpy arrays representing sequences of consecutive equal values.
    """
    # Find the indices where the values change
    change_indices = np.where(np.diff(arr) != 0)[0] + 1

    # Split the array into subarrays at the change indices
    sequences = np.split(arr, change_indices)

    # Remove empty subarrays
    sequences = [seq for seq in sequences if len(seq) > 0]

    return sequences


class SpeechRecorder:
    """
    Records speech from microphone
    """

    def __init__(
            self,
            vad: VAD,
            sample_rate: int=16000,
            seconds_per_block: float=2,
            min_speech_seconds: float=0.3,
            max_silence_seconds: float=0.6,
        ):
        """
        Initializes SpeechRecorder

        Args:
            vad (VAD): Voice Activity Detector
            sample_rate (int, optional): Sample rate. Defaults to 16000.
            seconds_per_block (float, optional): Seconds per block. Defaults to 1.
                A block is the unit of audio data that is sent for voice activity detection.
            min_speech_seconds (float, optional): Minimum speech seconds. Defaults to 0.5.
            max_silence_seconds (float, optional): Maximum silence seconds. Defaults to 0.5.
        """

        self.vad = vad
        self.sample_rate = sample_rate
        self.seconds_per_block = seconds_per_block
        self.min_speech_seconds = min_speech_seconds
        self.max_silence_seconds = max_silence_seconds
        self.backlog_audio = []
        self.recording = False
        self.recording_thread = None
    
    def record_audio_thread_fn(self):
        """
        Records audio from microphone
        """
        self.backlog_audio = []
        while self.recording:
            audio = sounddevice.rec(
                int(self.seconds_per_block * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                blocking=True,
                dtype='float32'
            )
            audio = audio.flatten()
            self.backlog_audio.append(audio)
    
    def generate_audio(self) -> Generator[np.ndarray, None, None]:
        """
        Generates audio from microphone
        """
        self.recording = True
        self.recording_thread = threading.Thread(target=self.record_audio_thread_fn)
        self.recording_thread.start()
        current_speech = []
        leftover_distance = 0
        try:
            while self.recording:
                if len(self.backlog_audio) > 0:
                    audio = self.backlog_audio.pop(0)
                    vad_signal = self.vad.detect(audio, self.sample_rate)

                    if np.sum(vad_signal) == 0:
                        if current_speech and len(np.concatenate(current_speech)) / self.sample_rate >= self.min_speech_seconds:
                            yield np.concatenate(current_speech)
                            current_speech = []
                        else:
                            current_speech = []
                            continue
                    else:
                        sequences = get_consecutive_sequences(vad_signal)
                        last_was_speech = False
                        last_speech_idx = -1
                        del_idxs = []
                        for i, sequence in enumerate(sequences):
                            if sequence[0] == 0 and i == len(sequences) - 1:
                                # Don't join the last sequence if it's silence
                                continue

                            if sequence[0] == 0 and last_was_speech and len(sequence) < self.max_silence_seconds * self.sample_rate:
                                del_idxs.append(i - len(del_idxs))
                                sequences[last_speech_idx] = np.concatenate([sequences[last_speech_idx], np.ones(len(sequence))])
                                continue
                            elif sequence[0] == 0:
                                last_was_speech = False
                                continue
                            if sequence[0] == 1:
                                if last_was_speech:
                                    sequences[last_speech_idx] = np.concatenate([sequences[last_speech_idx], sequence])
                                    del_idxs.append(i - len(del_idxs))
                                else:
                                    last_was_speech = True
                                    last_speech_idx = i
                        
                        for i in del_idxs:
                            del sequences[i]

                        audio_sequences = []
                        cursor = 0
                        for sequence in sequences:
                            audio_sequences.append(audio[cursor:cursor + len(sequence)])
                            cursor += len(sequence)
                        
                        if len(sequences) == 1:  # Must contain only speech as we checked for all silence
                            current_speech.append(audio_sequences[0])
                            leftover_distance = 0
                            continue
                            
                        if current_speech and sequences[0][0] == 0:
                            if len(sequences[0]) + leftover_distance >= self.min_speech_seconds * self.sample_rate:
                                current_speech.append(audio_sequences[0])
                                yield np.concatenate(current_speech)
                                current_speech = []
                                leftover_distance = 0
                                del sequences[0]
                                del audio_sequences[0]
                            elif len(sequences) > 3:
                                current_speech.append(audio_sequences[0])
                                current_speech.append(audio_sequences[1])
                                if len(np.concatenate(current_speech) > self.min_speech_seconds * self.sample_rate):
                                    yield np.concatenate(current_speech)
                                current_speech = []
                                leftover_distance = 0
                                del sequences[0]
                                del sequences[1]
                                del audio_sequences[0]
                                del audio_sequences[1]
                            elif len(sequences) == 3:
                                current_speech.append(audio_sequences[0])
                                current_speech.append(audio_sequences[1])
                                if len(sequences[-1]) > self.max_silence_seconds * self.sample_rate:
                                    if len(np.concatenate(current_speech) > self.min_speech_seconds * self.sample_rate):
                                        yield np.concatenate(current_speech)
                                    current_speech = []
                                    leftover_distance = 0
                                    continue
                                else:
                                    current_speech.append(audio_sequences[-1])
                                    leftover_distance = len(sequences[-1])
                                    continue
                            else:
                                current_speech.append(audio_sequences[0])
                                current_speech.append(audio_sequences[1])
                                leftover_distance = 0
                                continue
                        elif current_speech and sequences[0][0] == 1:
                            if len(sequences) == 2 and len(sequences[-1]) < self.max_silence_seconds * self.sample_rate:
                                current_speech.append(audio_sequences[0])
                                current_speech.append(audio_sequences[1])
                                leftover_distance = len(sequences[-1])
                                continue
                            else:
                                current_speech.append(audio_sequences[0])
                                if len(np.concatenate(current_speech)) > self.min_speech_seconds * self.sample_rate:
                                    yield np.concatenate(current_speech)
                        
                        if len(sequences) == 1 and sequences[0][0] == 1:
                            current_speech = [audio_sequences[0]]
                            leftover_distance = 0
                            continue
                        elif len(sequences) >= 2 and sequences[-1][0] == 1:
                            current_speech = [audio_sequences[-1]]
                            leftover_distance = 0
                            del sequences[-1]
                            del audio_sequences[-1]
                        elif len(sequences) >= 2 and sequences[-1][0] == 0 and len(sequences[-1]) < self.max_silence_seconds * self.sample_rate:
                            current_speech = [audio_sequences[-2], audio_sequences[-1]]
                            leftover_distance = len(sequences[-1])
                            del sequences[-1]
                            del sequences[-1]
                            del audio_sequences[-1]
                            del audio_sequences[-1]
                        
                        for i, sequence in enumerate(sequences):
                            if sequence[0] == 1 and len(sequence) > self.min_speech_seconds * self.sample_rate:
                                yield audio_sequences[i]
        except KeyboardInterrupt:
            self.stop()
            raise
            
    def stop(self):
        self.recording = False
        self.recording_thread.join()
    


if __name__ == '__main__':
    import os
    import soundfile as sf
    from .voice_activity_dection.xgb import XGBVAD

    if not os.path.exists('debug'):
        os.mkdir('debug')

    vad = XGBVAD()

    recorder = SpeechRecorder(vad, min_speech_seconds=0.5, max_silence_seconds=0.5)
    try:
        for i, segment in enumerate(recorder.generate_audio()):
            sf.write(f'debug/{i}.wav', segment, 16000)
            if i > 10:
                recorder.stop()
    except KeyboardInterrupt:
        recorder.stop()
        raise
