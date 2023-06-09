import os
import openai

try:
    with open('.env') as f:
        for line in f:
            key, value = line.strip().split('=')
            os.environ[key] = value
except FileNotFoundError:
    raise FileNotFoundError('Please create a .env file in the root directory of the project.')
openai.api_key = os.getenv("OPENAI_API_KEY")


def query_chatgpt(input_text: str) -> str:
    """
    Queries the GPT-3.5 Turbo model with the given input text and returns the
    generated response.

    Args:
        input_text (str): The input text to query the model with.
    
    Returns:
        str: The generated response.
    """
    completions = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": input_text}],
    )
    return completions.choices[0].message.content
