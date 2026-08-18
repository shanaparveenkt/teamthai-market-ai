import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    return OpenAI(api_key=api_key)


def generate_response(prompt, system_prompt):

    client = get_client()

    model = os.getenv(
        "OPENAI_MODEL",
        "gpt-5.6-luna"
    )

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=prompt
    )

    return response.output_text