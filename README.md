# Speech & Audio Project

This repo contains my Speech and Audio Project submission. This project is a chatGPT voice assistant.
Similar to the likes of Alex/Siri/Google Assistant, this voice assistant is able to answer questions, tell jokes, and
provide general assistance to the user, through the use of ASR, TTS and querying the ChatGPT API.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the program, run the following command:

```bash
python -m source.app
```

Before doing this however, you will need to create a .env file, matching the template and supply an API key for the
ChatGPT API.

## Exploratory Directory

The exploratory directory contains notebooks and data that I used in building this application.
Of particular interest is the `Voice-Activity-Detection.ipynb` notebook, which contains the code
for the VAD model that I used in the application.