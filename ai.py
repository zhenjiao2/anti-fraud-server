"""Run this model in Python

> pip install azure-ai-inference
"""
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.ai.inference.models import ImageContentItem, ImageUrl, TextContentItem
from azure.core.credentials import AzureKeyCredential

# 调用AI大模型来判断message的可疑程度
# 会返回一个如下面格式的数据ai_response，包括可疑程度，原因和建议
#```json 
# { 
#   "suspect_level": 5, 
#   "reasoning": "对方自称银行职员要求提供身份证号和验证码，这是敏感信息，且未明确说明账户异常的具体情况。", 
#   "advice": "立即挂断电话，不要提供任何信息，联系银行官方客服号码核实情况。" } 
# ```        
def check_suspect_level(message):
    # 为了安全起见，我们将敏感信息存储在环境变量中，在这里读取环境变量，包括如下变量
    # 模型的网址：MODEL_ENDPOINT
    # 访问密钥：GITHUB_TOKEN
    # api版本：MODEL_API_VERSION
    # 模型的名字：MODEL_NAME
    # 系统提示：MODEL_SYS_PROMPT
    # 用户提示：MODEL_USER_PROMPT
    load_dotenv()

    # 创建一个访问大模型的客户端
    client = ChatCompletionsClient(
        endpoint = os.environ["MODEL_ENDPOINT"],
        credential = AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
        api_version = os.environ["MODEL_API_VERSION"],
    )

    # 调用客户端的complete API来调用大模型
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

    # 将大模型的回答返回
    return response.choices[0].message.content