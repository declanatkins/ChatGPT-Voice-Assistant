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

When running the assistant will wait for you to say "hello", it will then respond and start listening for your query. Once you stop talking the query recording will complete. The assistant will then query the ChatGPT API and respond with the answer.
Once you are finished with the app, it will wait for you to say "finish", at which point it will stop listening and terminate.

Before doing this however, you will need to create a .env file, matching the template and supply an API key for the
ChatGPT API.

## Exploratory Directory

The exploratory directory contains notebooks and data that I used in building this application.
To run these you will first have to download some data using the following command:

```bash
python exploratory/download_files.py
```