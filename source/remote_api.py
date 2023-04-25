"""As I don't have a dedicated GPU on my Mac, I've created this file to run the model inference
on a remote server. This is a simple flask app that exposes endpoints for inference on each of the
models. The app is deployed on a remote server using gunicorn.
"""
from io import BytesIO
from flask import Flask, request, jsonify, send_file
import numpy as np
import soundfile as sf
import wave
from speech_to_text.pretrained_vosk import PretrainedVoskTranscriber
from speech_to_text.pretrained_whisper import PretrainedWhisperTranscriber
from text_to_speech.pretrained_coqui import PretrainedCoquiSynthesizer


app = Flask(__name__)
vosk = PretrainedVoskTranscriber()
whisper = PretrainedWhisperTranscriber()
coqui = PretrainedCoquiSynthesizer()


@app.route('/transcribe/vosk', methods=['POST'])
def transcribe_vosk():
    audio_file = request.files['audio']
    audio_file.save('/tmp/audio.wav')
    with wave.open('/tmp/audio.wav', 'rb') as wav:
        audio_data = wav.readframes(wav.getnframes())
    text = vosk.transcribe(audio_data)
    return jsonify({'text': text})


@app.route('/transcribe/whisper', methods=['POST'])
def transcribe_whisper():
    audio_file = request.files['audio']
    audio_file.save('/tmp/audio.wav')
    y, _ = sf.read('/tmp/audio.wav', dtype='float32')
    text = whisper.transcribe(y)
    return jsonify({'text': text})


@app.route('/synthesize/coqui', methods=['POST'])
def synthesize_coqui():
    text = request.json['text']
    audio_data = coqui.synthesize(text)

    buffer = BytesIO()
    np.save(buffer, audio_data)
    return send_file(buffer, mimetype='audio/wav')
