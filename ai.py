"""Run this model in Python

> pip install azure-ai-inference
"""
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.ai.inference.models import ImageContentItem, ImageUrl, TextContentItem
from azure.core.credentials import AzureKeyCredential

def check_suspect_level(message):
    load_dotenv()

    client = ChatCompletionsClient(
        endpoint = os.environ["MODEL_ENDPOINT"],
        credential = AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
        api_version = os.environ["MODEL_API_VERSION"],
    )

    client = ChatCompletionsClient(
        endpoint = os.environ["MODEL_ENDPOINT"],
        credential = AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
        api_version = os.environ["MODEL_API_VERSION"],
    )

    print(os.environ["MODEL_SYS_PROMPT"])
    print(os.environ["MODEL_USER_PROMPT"])
    response = client.complete(
        messages = [
            SystemMessage(content = os.environ["MODEL_SYS_PROMPT"]),
            UserMessage(content = [
                TextContentItem(text = f"{os.environ["MODEL_USER_PROMPT"]} {message}"),
            ]),
        ],
        model = os.environ["MODEL_NAME"],
        response_format = "text",
        max_tokens = 4096,
        temperature = 1,
        top_p = 1,
    )

    return response.choices[0].message.content