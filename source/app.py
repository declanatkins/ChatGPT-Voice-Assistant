from enum import Enum
import numpy as np
import pyaudio
from .speech_to_text.abc import Transcriber
from .text_to_speech.abc import Synthesizer


from scipy.io.wavfile import write


class AssistantState(Enum):
    """Enum for the assistant state.
    """
    RECORDING_FOR_VOICE_ACTIVATION = 0
    RECORDING_FOR_SPEECH_TO_TEXT = 1


class VoiceAssistantApplication:
    """Class for the voice assistant application.
    """

    def __init__(
            self,
            transcriber: Transcriber,
            synthesizer: Synthesizer,
            sampling_rate: int=16000,
            activation_phrase: str="Hey bot",
            block_size_seconds: float=1.5,
        ):
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.sampling_rate = sampling_rate
        self.state = AssistantState.RECORDING_FOR_VOICE_ACTIVATION
        self.activation_phrase = activation_phrase
        self.block_size_seconds = block_size_seconds
        self.device_index = self._set_recording_device()

    def _set_recording_device(self):
        """Sets the recording device.
        """
        audio = pyaudio.PyAudio()
        for i in range(audio.get_device_count()):
            dev = audio.get_device_info_by_index(i)
            print(f'{i} - {dev["name"]}')
        
        device_index = int(input("Enter device index: "))
        return device_index
    
    def audio_chunk_generator(self):
        """Yields audio chunks from the microphone.
        
        Yields:
            np.ndarray: The audio chunk.
        """

        try:
            # Create pyaudio instance
            audio = pyaudio.PyAudio()

            # Open stream
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sampling_rate,
                input=True,
                frames_per_buffer=1024,
                input_device_index=self.device_index,
            )

            # Yield audio chunks of size self.block_size_seconds
            buffers = []
            while True:
                audio_chunk = np.frombuffer(stream.read(1024), dtype=np.int16)
                buffers.append(audio_chunk)
                
                if len(buffers) >= int(self.block_size_seconds * self.sampling_rate / 1024):
                    yield np.concatenate(buffers)
                    buffers = []
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def run(self):
        """Runs the voice assistant application.
        """
        print("Voice assistant application is running...")

        # Chunk of audio data that contains speech data
        active_chunks = []

        # Number of iterations to wait for speech to be present
        max_wait_iterations = 10
        curr_wait_iterations = 0
        c = 0
        while True:
            print('Recording...')
            audio_chunk = next(self.audio_chunk_generator())

            write(f'audio_{c}.wav', 16000, audio_chunk)
            c += 1

            if self.state == AssistantState.RECORDING_FOR_VOICE_ACTIVATION:
                # If speech is present, add to active chunks
                if check_speech_present(audio_chunk, self.sampling_rate):
                    print('Active speech detected. Recording for voice activation...')
                    active_chunks.append(audio_chunk)
                elif active_chunks:
                    audio_data = np.concatenate(active_chunks)
                    print('Transcribing audio for activation phrase...')
                    text = self.transcriber.transcribe(audio_data).lower()
                    if self.activation_phrase.lower() in text:
                        self.state = AssistantState.RECORDING_FOR_SPEECH_TO_TEXT
                        active_chunks = []
                        print('Activation phrase detected. Recording for speech to text...')
                    else:
                        active_chunks = []
                        print(f'Detected Phrase was: {text}')
                        print('Activation phrase not detected. Recording for voice activation...')
                else:
                    print('No speech detected. Recording for voice activation...')
                    active_chunks = []
            elif self.state == AssistantState.RECORDING_FOR_SPEECH_TO_TEXT:
                if not check_speech_present(audio_chunk, self.sampling_rate) and active_chunks:
                    self.state = AssistantState.RECORDING_FOR_VOICE_ACTIVATION
                    audio_data = np.concatenate(active_chunks)
                    active_chunks = []
                    text = self.transcriber.transcribe(audio_data)
                    response = query_chatgpt(text)
                    self.synthesizer.synthesize(response)
                    print(f"Assistant: {response}")

                    # Play the audio
                    audio = pyaudio.PyAudio()
                    stream = audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sampling_rate,
                        output=True,
                    )
                    stream.write(self.synthesizer.synthesize(response))
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()


                elif not check_speech_present(audio_chunk, self.voice_activation_threshold):
                    # Wait for speech to be present for a few iterations
                    curr_wait_iterations += 1
                    if curr_wait_iterations >= max_wait_iterations:
                        curr_wait_iterations = 0
                        self.state = AssistantState.RECORDING_FOR_VOICE_ACTIVATION
                        print('No speech detected. Recording for voice activation...')
                else:
                    active_chunks.append(audio_chunk)
                    curr_wait_iterations = 0


if __name__ == '__main__':
    from .speech_to_text.pretrained_vosk import PretrainedVoskTranscriber
    from .text_to_speech.pretrained_coqui import PretrainedCoquiSynthesizer
    
    transcriber = PretrainedVoskTranscriber()
    synthesizer = PretrainedCoquiSynthesizer()

    app = VoiceAssistantApplication(transcriber, synthesizer)
    app.run()
