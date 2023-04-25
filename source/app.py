from typing import Union
import numpy as np
import soundfile as sf
import wave
from .recorder import SpeechRecorder
from .speech_to_text import AudioDataType, Transcriber
from .text_to_speech import Synthesizer
from .voice_activity_dection import VAD
from .util import query_chatgpt


ACTIVATION_PHRASE = 'Hello. How can I help you?'
CLOSING_PHRASE = 'Goodbye.'


class AssistantApp:
    """Voice Assistant Application
    """

    def __init__(
            self,
            transcriber: Transcriber,
            synthesizer: Synthesizer,
            vad: VAD,
            sample_rate: int=16000,
            seconds_per_block: int=2,
            activation_keyword: str="hello",
            closing_keyword: str="finish",

    ):
        """Initializes AssistantApp
        
        Args:
            transcriber (Transcriber): Transcriber
            synthesizer (Synthesizer): Synthesizer
            vad (VAD): Voice Activity Detector
            audio_data_type (AudioDataType): Audio data type
            sample_rate (int, optional): Sample rate. Defaults to 16000.
            chunk_size (int, optional): Chunk size. Defaults to 1024.
            seconds_per_block (int, optional): Seconds per block. Defaults to 1.
                A block is the unit of audio data that is sent for voice activity detection.
        """

        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.vad = vad
        self.sample_rate = sample_rate
        self.activation_keyword = activation_keyword
        self.closing_keyword = closing_keyword

        self.recorder = SpeechRecorder(
            vad=vad,
            sample_rate=sample_rate,
            seconds_per_block=seconds_per_block,
        )

    def _convert_audio_type(self, audio: np.ndarray) -> Union[np.ndarray, bytes]:
        if self.transcriber.audio_data_type == AudioDataType.NUMPY:
            return audio
        elif self.transcriber.audio_data_type == AudioDataType.BYTES:
            sf.write('/tmp/audio.wav', audio, self.sample_rate)
            with wave.open('/tmp/audio.wav', 'rb') as wav:
                return wav.readframes(wav.getnframes())
        else:
            raise ValueError(f'Unknown audio type: {self.transcriber.audio_type}')

    def wait_for_activation(self):
        """Waits for activation keyword using microphone chunks
        """
        print('Waiting for activation...')
        for speech_phrase in self.recorder.generate_audio():
            speech_phrase_text = self.transcriber.transcribe(self._convert_audio_type(speech_phrase))
            print(f'Heard: {speech_phrase_text}')
            if self.activation_keyword in speech_phrase_text.lower():
                break
            if self.closing_keyword in speech_phrase_text.lower():
                self.recorder.stop()
                self.synthesizer.say(CLOSING_PHRASE)
                exit(0)
        self.recorder.stop()
        self.synthesizer.say(ACTIVATION_PHRASE)
        return True

    def get_speech_query(self):
        """Gets speech query from user
        """
        print('Recording query...')
        speech_query = next(self.recorder.generate_audio())
        print('Query recorded')
        self.recorder.stop()
        query_text = self.transcriber.transcribe(self._convert_audio_type(speech_query))
        return query_text  

    def run(self):
        """Main loop
        """
        while True:
            self.wait_for_activation()
            query = self.get_speech_query()
            if query:
                print(f'Query was: {query}')
                chat_gpt_response = query_chatgpt(query)
                print(f'ChatGPT response: {chat_gpt_response}')
                self.synthesizer.say(chat_gpt_response)
            else:
                self.synthesizer.say("Sorry, I didn't catch that")


if __name__ == "__main__":
    from .voice_activity_dection.xgb import XGBVAD
    from .voice_activity_dection.energy_thresholding import EnergyThresholdingVAD
    from .speech_to_text.pretrained_whisper import PretrainedWhisperTranscriber
    from .text_to_speech.pretrained_coqui import PretrainedCoquiSynthesizer
    from .text_to_speech.remote_coqui import RemoteCoquiSynthesizer
    from .speech_to_text.remote_whisper import RemoteWhisperTranscriber

    vad = XGBVAD()
    # vad = EnergyThresholdingVAD()
    transcriber = RemoteWhisperTranscriber('http://0.0.0.0:8000/transcribe/whisper')
    synthesizer = RemoteCoquiSynthesizer('http://0.0.0.0:8000/synthesize/coqui')

    assistant = AssistantApp(transcriber, synthesizer, vad)
    assistant.run()
