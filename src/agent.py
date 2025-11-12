"""
严格使用LangChain和Workflow的反欺诈分析系统
基于agent_langchain.py和agent_workflow.py的内容，实现完整的工作流集成

Requirements:
> pip install langchain-openai langchain-core azure-ai-inference beautifulsoup4 requests mcp selenium httpx
> 下载ChromeDriver: https://chromedriver.chromium.org/
> python agent_work&lang.py

注意: 
- selenium用于支持JavaScript渲染的网页
- 需要安装Chrome浏览器和对应版本的ChromeDriver
- 如果没有selenium，程序会回退到普通的requests方式
"""

import asyncio
import json
import os
import requests
import concurrent.futures
import logging
from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack
from urllib.parse import urlparse
from pathlib import Path
import argparse

import httpx

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from bs4 import BeautifulSoup

# Selenium导入（用于JavaScript支持）

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


# MCP导入（用于外部工具集成）
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Azure AI Inference导入
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage, 
    SystemMessage as AzureSystemMessage, 
    UserMessage, 
    ToolMessage as AzureToolMessage
)
from azure.core.credentials import AzureKeyCredential


# 全局日志器
logger = logging.getLogger(__name__)


def setup_logging(verbose_level: int = 1):
    """
    设置日志级别
    
    Args:
        verbose_level: 详细程度级别
            0: 只输出错误信息
            1: 输出错误和重要信息（默认）
            2: 输出详细信息
            3: 输出调试信息
    """
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 设置日志级别
    level_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG
    }
    
    log_level = level_map.get(verbose_level, logging.WARNING)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 创建格式器
    if verbose_level >= 3:
        # 调试模式显示详细信息
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        # 简化格式
        formatter = logging.Formatter('%(message)s')
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(log_level)
    
    # 设置根日志器级别
    logging.getLogger().setLevel(log_level)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则尝试默认位置
    
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = Path("config.json")
    else:
        config_path = Path(config_path)
    
    config = None
    
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"[Config] 成功加载配置文件: {config_path}")
    except Exception as e:
        logger.error(f"[Config] 读取配置文件失败 {config_path}: {str(e)}")
    
    if config is None:
        logger.warning("[Config] 未找到配置文件，使用默认配置")
        # 返回默认配置
        config = {
            "azure_ai_inference": {
                "endpoint": "https://models.github.ai/inference",
                "api_key": os.environ.get("GITHUB_TOKEN", ""),  # 将依赖环境变量
                "model_name": "gpt-4.1-mini",
                "temperature": 0.3,
                "top_p": 1.0,
                "max_tokens": 1000,
                "timeout": 20,
                "max_retries": 3,
                "payload": {
                    "model": "gpt-4.1-mini",
                    "temperature": 0.3,
                    "top_p": 1.0,
                    "max_tokens": 1000,
                    "stream": False,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            },
            "workflow": {
                "total_timeout": 120,
                "enable_mcp": False,
                "verbose_level": 1
            }
        }
    
    # 验证配置格式
    if "azure_ai_inference" not in config:
        config["azure_ai_inference"] = {}
    if "workflow" not in config:
        config["workflow"] = {}
    
    # 处理环境变量替换：支持 ${ENV_VAR} 语法
    def _resolve_env_vars(value):
        """递归解析配置值中的环境变量引用"""
        if isinstance(value, str):
            import re
            # 匹配 ${VAR_NAME} 格式
            pattern = r'\$\{([^}]+)\}'
            
            def replacer(match):
                var_name = match.group(1)
                env_value = os.environ.get(var_name)
                if env_value is None:
                    logger.warning(f"[Config] 环境变量 {var_name} 未设置，保持原值")
                    return match.group(0)
                return env_value
            
            return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: _resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve_env_vars(item) for item in value]
        else:
            return value
    
    # 解析整个配置中的环境变量
    config = _resolve_env_vars(config)
    
    # 设置日志级别（如果配置中有的话）
    verbose_level = config.get("workflow", {}).get("verbose_level", 1)
    setup_logging(verbose_level)
    
    return config


class AzureInferenceLangChainModel(BaseChatModel):
    """
    LangChain兼容的Azure AI Inference聊天模型
    严格继承LangChain的BaseChatModel
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        
        # 保存配置以供后续使用
        self._config = config or {}
        
        # 如果传入了配置，使用配置文件的参数，否则使用kwargs
        if config and "azure_ai_inference" in config:
            azure_config = config["azure_ai_inference"]
            logger.info(f"[Model] 使用配置文件中的Azure AI Inference设置")
        else:
            azure_config = {}
            logger.info(f"[Model] 使用默认或传入的参数")
        
        # 使用私有属性避免Pydantic字段冲突
        # kwargs中的参数优先级高于配置文件
        self._client = None
        self._endpoint = kwargs.get("endpoint", azure_config.get("endpoint", ""))
        self._api_key = kwargs.get("api_key", azure_config.get("api_key", ""))
        self._model_name = kwargs.get("model_name", azure_config.get("model_name", "gpt-4.1-mini"))
        self._temperature = kwargs.get("temperature", azure_config.get("temperature", 0.3))
        self._top_p = kwargs.get("top_p", azure_config.get("top_p", 1.0))
        self._max_tokens = kwargs.get("max_tokens", azure_config.get("max_tokens", 1000))
        self._timeout = kwargs.get("timeout", azure_config.get("timeout", 20))
        self._max_retries = kwargs.get("max_retries", azure_config.get("max_retries", 3))
        
        logger.info(f"[Model Config] 模型配置:")
        logger.info(f"  - 端点: {self._endpoint}")
        logger.info(f"  - 模型: {self._model_name}")
        logger.info(f"  - 温度: {self._temperature}")
        logger.info(f"  - Top-P: {self._top_p}")
        logger.info(f"  - 最大令牌: {self._max_tokens}")
        logger.info(f"  - 超时时间: {self._timeout}秒")
        logger.info(f"  - 最大重试: {self._max_retries}次")
        
        self._setup_client()
    
    @property
    def client(self):
        return self._client
    
    @property
    def endpoint(self):
        return self._endpoint
    
    @property
    def api_key(self):
        return self._api_key
    
    @property
    def model_name(self):
        return self._model_name
    
    @property
    def temperature(self):
        return self._temperature
    
    @property
    def top_p(self):
        return self._top_p
    
    @property
    def max_tokens(self):
        return self._max_tokens
    
    @property
    def timeout(self):
        return self._timeout
    
    @property
    def max_retries(self):
        return self._max_retries
    
    class Config:
        arbitrary_types_allowed = True
    
    def _setup_client(self):
        """设置Azure AI Inference客户端"""
        try:
            # 优先使用配置文件中的API密钥，如果没有则使用环境变量
            api_key = self._api_key
            if not api_key:
                raise ValueError("API密钥未配置: 请在config.json中设置api_key")
            
            # 检查端点是否配置
            if not self._endpoint:
                raise ValueError("端点未配置: 请在config.json中设置endpoint")
            
            # 创建客户端
            self._client = ChatCompletionsClient(
                endpoint=self._endpoint,
                credential=AzureKeyCredential(api_key),
            )
            
            logger.info(f"[Client] Azure AI Inference客户端初始化成功")
            logger.debug(f"[Client] 使用端点: {self._endpoint}")
            
        except Exception as e:
            error_msg = f"客户端初始化失败: {str(e)}"
            logger.error(f"[Client Error] {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _generate(
        self,
        messages: List[Any],
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        """使用Azure AI Inference生成聊天回复"""
        # 转换LangChain消息格式到Azure格式
        azure_messages = self._convert_to_azure_messages(messages)
        
        # 获取可用工具
        tools = kwargs.get('tools', [])
        
        try:
            logger.debug(f"[Model] 正在调用模型: {self.model_name}")
            
            # 使用线程池执行器来强制超时
            def _sync_call():
                return self.client.complete(
                    messages=azure_messages,
                    model=self.model_name,
                    tools=tools,
                    response_format="text",
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )
            
            # 在单独线程中执行，设置强制超时
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_sync_call)
                try:
                    # 设置20秒强制超时
                    response = future.result(timeout=20)
                except concurrent.futures.TimeoutError:
                    raise Exception("模型调用超时(20秒)")
            
            # 检查响应是否有效
            if not response or not response.choices:
                raise Exception("模型返回空响应或无选择项")
            
            # 转换响应回LangChain格式
            choice = response.choices[0]
            message = choice.message
            
            if not message:
                raise Exception("模型返回的消息为空")
            
            logger.debug(f"[Model] 模型调用成功")
            
            if message.tool_calls:
                # 处理工具调用
                ai_message = AIMessage(
                    content=message.content or "",
                    tool_calls=[{
                        "name": tool_call.function.name,
                        "args": json.loads(tool_call.function.arguments),
                        "id": tool_call.id
                    } for tool_call in message.tool_calls]
                )
            else:
                ai_message = AIMessage(content=message.content)
            
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
            
        except Exception as e:
            # 模型调用失败时输出错误信息并抛出异常
            error_msg = f"模型调用失败: {str(e)}"
            logger.error(f"[Model Error] {error_msg}")
            raise RuntimeError(error_msg) from e
    
    async def _agenerate(
        self,
        messages: List[Any],
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步版本的模型调用，使用httpx进行真正的异步HTTP调用"""
        try:
            logger.debug(f"[Model] 异步调用模型: {self.model_name}")
            
            # 如果httpx可用，使用真正的异步HTTP调用
            return await self._async_http_call(messages, **kwargs)
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP状态错误: {e.response.status_code} - {e.response.text}"
            logger.error(f"[Model Error] {error_msg}")
            raise RuntimeError(error_msg) from e
        except httpx.TimeoutException as e:
            error_msg = f"HTTP请求超时: {str(e)}"
            logger.error(f"[Model Error] {error_msg}")
            raise RuntimeError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"HTTP请求错误: {str(e)}"
            logger.error(f"[Model Error] {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"异步模型调用失败: {type(e).__name__}: {str(e)}"
            if not str(e):  # 如果异常消息为空，尝试获取更多信息
                error_msg += f" (异常类型: {type(e)}, 异常属性: {dir(e)})"
            logger.error(f"[Model Error] {error_msg}")
            raise RuntimeError(error_msg) from e
    
    async def _async_http_call(self, messages: List[Any], **kwargs: Any) -> ChatResult:
        """使用httpx进行真正的异步HTTP调用"""
        import httpx
        
        # 直接从LangChain消息转换为API格式，跳过Azure消息对象
        tools = kwargs.get('tools', [])
        
        # 直接转换LangChain消息为标准API格式
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # 处理工具调用
                    tool_calls = [{
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["args"])
                        }
                    } for tool_call in msg.tool_calls]
                    api_messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": tool_calls
                    })
                else:
                    api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": str(msg.content)
                })
        
        # 准备请求数据 - 从配置文件的payload部分获取参数
        azure_config = getattr(self, '_config', {}).get("azure_ai_inference", {})
        config_payload = azure_config.get("payload", {})
        
        # 基础请求结构
        payload = {
            "messages": api_messages
        }
        
        if config_payload:
            # 如果配置文件中有payload，直接使用所有配置的参数
            for key, value in config_payload.items():
                payload[key] = value
            logger.debug(f"[Model] 使用配置文件中的payload参数: {list(config_payload.keys())}")
        else:
            # 如果没有配置payload，使用实例属性作为默认值
            payload.update({
                "model": self.model_name,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            })
            logger.debug(f"[Model] 配置文件中未找到payload，使用默认参数: {list(payload.keys())}")
        
        # 添加工具（如果有）
        if tools:
            payload["tools"] = tools
        
        # 设置请求头 - 使用配置文件中的API密钥
        api_key = self.api_key
        if not api_key:
            raise ValueError("API密钥未配置")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 使用httpx进行真正的异步HTTP调用
        timeout_seconds = self.timeout
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            logger.debug(f"[Model] 发送异步HTTP请求...")
            logger.debug(f"[Model] 请求URL: {self.endpoint}")
            logger.debug(f"[Model] 超时时间: {timeout_seconds}秒")
            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"[Model] 请求头: {headers}")
                logger.debug(f"[Model] 请求数据: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            
            try:
                response = await client.post(
                    f"{self.endpoint}",
                    json=payload,
                    headers=headers,
                    timeout=timeout_seconds
                )
                
                logger.debug(f"[Model] 收到响应，状态码: {response.status_code}")
                
                # 检查响应状态
                if response.status_code != 200:
                    error_text = response.text
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {error_text}",
                        request=response.request,
                        response=response
                    )
                
                result_data = response.json()
                logger.debug(f"[Model] 响应解析成功")
                
            except httpx.HTTPStatusError:
                raise  # 重新抛出HTTP状态错误
            except Exception as e:
                logger.error(f"[Model] HTTP请求过程中发生错误: {type(e).__name__}: {str(e)}")
                raise
        
        logger.debug(f"[Model] 异步HTTP调用成功")
        
        # 检查响应是否有效
        if not result_data.get("choices"):
            raise Exception("模型返回空响应或无选择项")
        
        # 解析响应
        choice = result_data["choices"][0]
        message_content = choice["message"]["content"]
        
        if not message_content:
            raise Exception("模型返回的消息为空")
        
        # 构造LangChain响应
        ai_message = AIMessage(content=message_content)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])
    
    def _convert_to_azure_messages(self, messages: List[Any]) -> List[Any]:
        """转换LangChain消息到Azure AI Inference格式"""
        azure_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                azure_messages.append(AzureSystemMessage(content=msg.content))
            elif isinstance(msg, HumanMessage):
                azure_messages.append(UserMessage(content=msg.content))
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    azure_messages.append(AssistantMessage(
                        tool_calls=[{
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"])
                            }
                        } for tool_call in msg.tool_calls]
                    ))
                else:
                    azure_messages.append(AssistantMessage(content=msg.content))
            elif isinstance(msg, ToolMessage):
                azure_messages.append(AzureToolMessage(
                    tool_call_id=msg.tool_call_id,
                    content=str(msg.content)
                ))
        
        return azure_messages
    
    @property
    def _llm_type(self) -> str:
        return "azure_inference_langchain"


class URLExtractorTool(BaseTool):
    """LangChain工具：从文本中提取URL链接"""
    
    name: str = "url_extractor"
    description: str = "从输入文本中提取所有URL链接，包括完整URL和潜在的短链接"
    
    def _run(self, text: str) -> str:
        """提取文本中的URL"""
        import re
        
        # 第一步：使用正则表达式提取所有可能的URL字符串
        all_potential_urls = []
        
        # 更全面的正则表达式模式，提取所有可能的URL
        url_patterns = [r'\b(https?://)?([-A-Za-z0-9_]+\.)?[-A-Za-z0-9_]+\.[-A-Za-z0-9_]{2,63}(/-A-Za-z0-9_*)*\b']
        
        # 提取所有匹配的URL字符串
        for pattern in url_patterns:
            # 使用 finditer 获取 Match 对象，避免因捕获组导致返回元组
            for m in re.finditer(pattern, text, re.IGNORECASE):
                full_match = m.group(0)
                # 清理URL末尾的无效字符
                cleaned_url = self._clean_url_end(full_match)
                if cleaned_url and len(cleaned_url) > 3:
                    all_potential_urls.append(cleaned_url)
        
        # 第二步：去重并按字符串长度从长到短排序
        unique_urls = list(set(all_potential_urls))
        unique_urls.sort(key=len, reverse=True)  # 按长度从长到短排序
        
        # 第三步：筛选出真正需要的URL（去除子字符串）
        result_urls = []
        
        for i, current_url in enumerate(unique_urls):
            is_substring = False
            
            # 检查当前URL是否是前面（更长的）URL的子字符串
            for j in range(i):
                longer_url = result_urls[j]
                if current_url in longer_url:
                    is_substring = True
                    break
            
            # 如果不是任何更长URL的子字符串，则添加到结果中
            if not is_substring:
                result_urls.append(current_url)
        
        # 第四步：标准化URL格式（添加协议前缀）
        standardized_urls = []
        for url in result_urls:
            if not url.startswith(('http://', 'https://')):
                if url.startswith('www.'):
                    standardized_urls.append('https://' + url)
                else:
                    standardized_urls.append('https://' + url)
            else:
                standardized_urls.append(url)
        
        
        return json.dumps({"extracted_urls": standardized_urls}, ensure_ascii=False)
    
    def _clean_url_end(self, url: str) -> str:
        """清理URL末尾的无效字符"""
        import re
        
        # 移除末尾的标点符号和中文字符
        # 保留URL中常见的有效字符
        valid_end_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/'
        
        # 从后往前找到最后一个有效字符的位置
        last_valid_pos = len(url)
        for i in range(len(url) - 1, -1, -1):
            if url[i] in valid_end_chars:
                last_valid_pos = i + 1
                break
            # 如果遇到中文字符，直接截断
            elif '\u4e00' <= url[i] <= '\u9fff':
                last_valid_pos = i
                break
        
        cleaned = url[:last_valid_pos]
        
        # 进一步清理：移除末尾的点号等
        cleaned = re.sub(r'[.,;!?。，；！？]+$', '', cleaned)
        
        return cleaned if len(cleaned) > 3 else ""  # 确保URL有足够长度
    
    async def _arun(self, text: str) -> str:
        """异步运行"""
        return self._run(text)


class WebContentFetcherTool(BaseTool):
    """LangChain工具：访问URL并获取网页内容"""
    
    name: str = "web_content_fetcher"
    description: str = "访问指定URL并获取网页内容，提取可读文本信息用于风险分析"
    
    def _run(self, url: str) -> str:
        """访问URL并获取内容，支持JavaScript渲染"""
        
        # 首先尝试使用Selenium（支持JavaScript）
        selenium_result = self._fetch_with_selenium(url)
        if selenium_result:
            return selenium_result
        
        # 如果Selenium失败或不可用，回退到requests
        return self._fetch_with_requests(url)
    
    def _fetch_with_selenium(self, url: str) -> str:
        """使用Selenium获取支持JavaScript的网页内容"""
        try:
            # 配置Chrome选项
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # 无头模式
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # 创建WebDriver
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(15)  # 设置页面加载超时
            
            try:
                logger.debug(f"[Selenium] 正在访问: {url}")
                driver.get(url)
                
                # 等待页面加载完成（等待body元素）
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                
                # 额外等待JavaScript执行
                driver.implicitly_wait(3)
                
                # 获取页面标题
                title = driver.title or "无标题"
                
                # 获取页面内容
                page_source = driver.page_source
                
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # 移除脚本和样式元素
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 获取主要文本内容
                text_content = soup.get_text()
                
                # 清理文本
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text_content = ' '.join(chunk for chunk in chunks if chunk)
                
                # 限制文本长度
                if len(text_content) > 2000:
                    text_content = text_content[:2000] + "..."
                
                result = {
                    "url": url,
                    "title": title,
                    "content": text_content,
                    "status": "success",
                    "method": "selenium"
                }
                
                logger.debug(f"[Selenium] 成功获取内容，标题: {title}")
                return json.dumps(result, ensure_ascii=False, indent=2)
                
            finally:
                driver.quit()
                
        except (TimeoutException, WebDriverException) as e:
            logger.warning(f"[Selenium] 访问失败: {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"[Selenium] 未知错误: {str(e)}")
            return None
    
    def _fetch_with_requests(self, url: str) -> str:
        """使用requests获取网页内容（备用方法）"""
        try:
            logger.debug(f"[Requests] 正在访问: {url}")
            # 设置请求头，模拟浏览器访问
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            # 发送HTTP请求
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            # 如果没有BeautifulSoup，使用简单的文本处理
            if BeautifulSoup is None:
                text_content = response.text
                title_text = "无法解析标题"
            else:
                # 解析HTML内容
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 移除脚本和样式元素
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 获取页面标题
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "无标题"
                
                # 获取主要文本内容
                text_content = soup.get_text()
                
                # 清理文本
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text_content = ' '.join(chunk for chunk in chunks if chunk)
            
            # 限制文本长度避免过长
            if len(text_content) > 2000:
                text_content = text_content[:2000] + "..."
            
            result = {
                "url": url,
                "title": title_text,
                "content": text_content,
                "status": "success",
                "method": "requests"
            }
            
            logger.debug(f"[Requests] 成功获取内容，标题: {title_text}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except requests.exceptions.RequestException as e:
            error_result = {
                "url": url,
                "error": f"访问失败: {str(e)}",
                "status": "error",
                "method": "requests"
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
        except Exception as e:
            error_result = {
                "url": url,
                "error": f"解析失败: {str(e)}",
                "status": "error",
                "method": "requests"
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
    
    async def _arun(self, url: str) -> str:
        """异步运行"""
        return self._run(url)


class MCPFetchTool(BaseTool):
    """LangChain工具：MCP工具包装器（用于外部fetch工具）"""
    
    name: str = "mcp_fetch"
    description: str = "使用MCP外部工具获取网页内容"
    
    def __init__(self, session: ClientSession, **kwargs):
        super().__init__(**kwargs)
        self.session = session
    
    def _run(self, url: str) -> str:
        """同步运行 - 在异步上下文中不使用"""
        raise NotImplementedError("请使用异步版本")
    
    async def _arun(self, url: str) -> str:
        """异步工具执行"""
        try:
            # 执行MCP工具
            result = await self.session.call_tool("fetch", {"url": url})
            logger.debug(f"[MCP工具 'fetch' 执行，URL: {url}]: 成功获取内容")
            return str(result.content)
            
        except Exception as e:
            error_msg = f"MCP工具执行失败: {str(e)}"
            logger.warning(error_msg)
            return error_msg


class LangChainFraudWorkflow(Runnable):
    """
    基于LangChain的反欺诈分析工作流
    严格继承LangChain的Runnable接口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化工作流"""
        # 初始化LangChain兼容的聊天模型，传入配置
        self.llm = AzureInferenceLangChainModel(config=config)
        
        # 保存配置用于工作流设置
        self.config = config or {}
        self.config = config
        
        # 初始化LangChain工具
        self.url_extractor = URLExtractorTool()
        self.web_fetcher = WebContentFetcherTool()
        
        # MCP工具相关
        self.exit_stack = AsyncExitStack()
        self.mcp_tools = []
        
        # 反欺诈分析的系统提示 - 从文件加载
        self.system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """从system_prompt.txt文件加载系统提示内容"""
        try:
            # 获取当前脚本所在目录
            current_dir = Path(__file__).parent
            prompt_file = current_dir / "system_prompt.txt"
            
            # 读取系统提示文件
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            logger.debug(f"[System Prompt] 成功从文件加载系统提示: {prompt_file}")
            return content
            
        except FileNotFoundError:
            # 如果文件不存在，返回默认的系统提示
            logger.warning(f"[System Prompt] 未找到system_prompt.txt文件，使用默认提示")
            return "你是一个反电信诈骗的专家，帮助用户分析对话内容并判断对方的诈骗嫌疑等级（1-5），并提供相应的对策建议。"
            
        except Exception as e:
            # 如果读取文件时出现其他错误，记录错误并返回默认提示
            logger.error(f"[System Prompt] 加载系统提示文件时出错: {str(e)}，使用默认提示")
            return "你是一个反电信诈骗的专家，帮助用户分析对话内容并判断对方的诈骗嫌疑等级（1-5），并提供相应的对策建议。"
    
    async def connect_mcp_server(self, server_id: str, command: str, args: List[str], env: Dict[str, str] = None):
        """连接到MCP服务器并注册工具"""
        if env is None:
            env = {}
            
        server_params = StdioServerParameters(command=command, args=args, env=env)
        
        try:
            # 连接到MCP服务器
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            
            # 列出可用工具并创建LangChain工具
            response = await session.list_tools()
            tools = response.tools
            
            # 为每个MCP工具创建LangChain工具
            for tool in tools:
                if tool.name == "fetch":
                    langchain_tool = MCPFetchTool(session=session)
                    self.mcp_tools.append(langchain_tool)
            
            logger.info(f"已连接到MCP服务器 '{server_id}'，包含工具: {[t.name for t in tools]}")
            return True
            
        except Exception as e:
            logger.error(f"连接MCP服务器失败: {str(e)}")
            return False
    
    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """LangChain Runnable接口的同步方法"""
        # 由于需要异步操作，这里抛出异常提示使用异步方法
        raise NotImplementedError("请使用 ainvoke 方法进行异步调用")
    
    async def ainvoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """LangChain Runnable接口的异步方法 - 运行完整工作流"""
        input_text = input.get("text", "")
        
        logger.info("开始LangChain反欺诈分析工作流...")
        logger.info(f"输入文本: {input_text}")
        
        workflow_result = {
            "input_text": input_text,
            "workflow_steps": [],
            "final_analysis": "",
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "status": "running"
        }
        
        try:
            # 为整个工作流设置总超时时间，从配置文件读取
            workflow_config = self.config.get("workflow", {})
            total_timeout = workflow_config.get("total_timeout", 180)
            logger.info(f"[Workflow] 开始执行，设置总超时时间为{total_timeout}秒")
            
            async def _run_workflow():
                # 步骤1: 初步语义分析
                logger.info("\n步骤1: 进行初步语义分析...")
                try:
                    initial_analysis = await self._step1_initial_analysis(input_text)
                    workflow_result["workflow_steps"].append({
                        "step": 1,
                        "name": "初步语义分析",
                        "result": initial_analysis,
                        "status": "success"
                    })
                    logger.info(f"初步分析完成")
                except Exception as e:
                    # 模型调用失败，立即停止工作流
                    workflow_result["status"] = "failed"
                    workflow_result["error"] = str(e)
                    workflow_result["failed_step"] = 1
                    logger.error(f"[CRITICAL] 步骤1失败，工作流停止: {str(e)}")
                    return workflow_result
                
                # 步骤2: 提取URL
                logger.info("\n步骤2: 提取URL链接...")
                extracted_urls = await self._step2_extract_urls(input_text)
                workflow_result["workflow_steps"].append({
                    "step": 2,
                    "name": "URL提取",
                    "result": extracted_urls,
                    "status": "success"
                })
                logger.debug(f"发现URL: {extracted_urls.get('extracted_urls', [])}")
                
                # 步骤3: 网页内容获取
                urls = extracted_urls.get('extracted_urls', [])
                web_contents = []
                if urls:
                    logger.info("\n步骤3: 获取网页内容...")
                    web_contents = await self._step3_fetch_web_content(urls)
                    workflow_result["workflow_steps"].append({
                        "step": 3,
                        "name": "网页内容获取",
                        "result": web_contents,
                        "status": "success"
                    })
                    logger.info(f"网页内容获取完成")
                else:
                    logger.info("\n未发现URL链接，跳过网页内容获取步骤")
                
                # 步骤4: 综合分析
                logger.info("\n步骤4: 进行综合反欺诈分析...")
                try:
                    final_analysis = await self._step4_comprehensive_analysis(
                        input_text, initial_analysis, web_contents
                    )
                    workflow_result["workflow_steps"].append({
                        "step": 4,
                        "name": "综合反欺诈分析",
                        "result": final_analysis,
                        "status": "success"
                    })
                    workflow_result["final_analysis"] = final_analysis
                    workflow_result["status"] = "completed"
                    logger.info("综合分析完成")
                except Exception as e:
                    # 模型调用失败，标记为失败但保留之前步骤的结果
                    workflow_result["status"] = "partial_failure"
                    workflow_result["error"] = str(e)
                    workflow_result["failed_step"] = 4
                    logger.error(f"[CRITICAL] 步骤4失败: {str(e)}")
                    
                return workflow_result
            
            # 在总超时控制下执行工作流
            result = await asyncio.wait_for(_run_workflow(), timeout=total_timeout)
            return result
            
        except asyncio.TimeoutError:
            # 工作流总超时
            error_msg = f"工作流执行超时({total_timeout}秒)"
            logger.error(f"\n[致命错误] {error_msg}")
            workflow_result["status"] = "timeout"
            workflow_result["error"] = error_msg
            workflow_result["stopped_at"] = "工作流总超时"
            return workflow_result
            
        except RuntimeError as e:
            # 模型调用失败，立即停止并输出错误信息
            error_msg = str(e)
            logger.error(f"\n[致命错误] 工作流已停止: {error_msg}")
            workflow_result["status"] = "model_failure"
            workflow_result["error"] = error_msg
            workflow_result["stopped_at"] = "模型调用失败"
            return workflow_result
        except Exception as e:
            # 其他类型的错误
            error_msg = f"工作流执行失败: {str(e)}"
            logger.error(f"\n[错误] {error_msg}")
            workflow_result["status"] = "error"
            workflow_result["error"] = error_msg
            return workflow_result
    
    async def _step1_initial_analysis(self, text: str) -> str:
        """步骤1: 使用LangChain进行初步语义分析"""
        messages = [
            SystemMessage(content="""你是一个专业的文本分析专家，负责对输入的文本进行多方面的深度分析。你的任务是识别以下内容：

# Instructions
1. **主要内容和目的**: 分析文本的主题和核心目的，并总结关键内容。
2. **可疑或异常表述**: 识别文本中是否存在可疑、不合逻辑或潜在误导的表述。
3. **发送者可能的身份或意图**: 根据文本内容推测撰写者可能的身份、背景或目的。
4. **关键信息点**: 提取文本中最重要的信息点，这些信息对于理解文本意图至关重要。
5. **诈骗信息的可能性分析**: 评估文本中包含诈骗意图或信息的可能性，并解释判断依据。

# Steps
请按照以下步骤完成分析：
1. 仔细阅读输入的文本，初步理解其主题和核心内容。
2. 对内容中的每一句话进行意图分析，标注可能异常或不合逻辑的地方。
3. 推测发送者的可能身份（如个人、公司、恶意诈骗者等）及其潜在动机。
4. 提取关键信息点，包括时间、地点、对象和具体事件等要素。
5. 评估文本中是否有诈骗信息的可能性，并详细说明原因（如语气、措辞、链接等特征）。
请用简洁明了的语言总结你的分析。"""),
            HumanMessage(content=f"请分析以下文本：\n\n{text}")
        ]
        
        try:
            logger.debug(f"[Step1] 开始异步模型调用...")
            result = await self.llm._agenerate(messages)
            
            # 检查结果是否为空
            if not result or not result.generations or not result.generations[0].message.content:
                raise RuntimeError("模型返回空结果")
                
            logger.debug(f"[Step1] 模型调用完成")
            return result.generations[0].message.content
            
        except Exception as e:
            error_msg = f"步骤1-初步分析失败: {str(e)}"
            logger.error(f"[Workflow Error] {error_msg}")
            raise RuntimeError(error_msg) from e
    
    async def _step2_extract_urls(self, text: str) -> Dict[str, List[str]]:
        """步骤2: 使用LangChain工具提取URL"""
        result = await self.url_extractor._arun(text)
        return json.loads(result)
    
    async def _step3_fetch_web_content(self, urls: List[str]) -> List[Dict]:
        """步骤3: 使用LangChain工具获取网页内容"""
        web_contents = []
        
        for url in urls:
            logger.debug(f"正在访问URL: {url}")
            
            # 优先尝试使用MCP工具
            content_result = None
            if self.mcp_tools:
                for mcp_tool in self.mcp_tools:
                    if mcp_tool.name == "mcp_fetch":
                        try:
                            content_result = await mcp_tool._arun(url)
                            break
                        except Exception as e:
                            logger.warning(f"MCP工具访问失败: {str(e)}，尝试内置工具")
            
            # 如果MCP工具失败，使用内置工具
            if content_result is None:
                content_result = await self.web_fetcher._arun(url)
            
            try:
                content_data = json.loads(content_result)
                web_contents.append(content_data)
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接使用文本内容
                web_contents.append({
                    "url": url,
                    "content": content_result,
                    "status": "success"
                })
        
        return web_contents
    
    async def _step4_comprehensive_analysis(self, original_text: str, initial_analysis: str, web_contents: List[Dict]) -> str:
        """步骤4: 使用LangChain进行综合反欺诈分析"""
        # 准备分析数据
        formatted_data = f"""
原始文本：
{original_text}

初步分析：
{initial_analysis}

网页内容分析：
"""
        
        for content in web_contents:
            if content.get("status") == "success":
                formatted_data += f"""
URL: {content.get('url', '未知')}
标题: {content.get('title', '无标题')}
内容摘要: {str(content.get('content', ''))[:500]}...
"""
            else:
                formatted_data += f"""
URL: {content.get('url', '未知')}
访问状态: 失败 - {content.get('error', '未知错误')}
"""
        
        # 进行最终分析
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"请分析以下内容：\n\n{formatted_data}")
        ]
        
        try:
            logger.debug(f"[Step4] 开始异步模型调用...")
            result = await self.llm._agenerate(messages)
            
            # 检查结果是否为空
            if not result or not result.generations or not result.generations[0].message.content:
                raise RuntimeError("模型返回空结果")
                
            logger.debug(f"[Step4] 模型调用完成")
            return result.generations[0].message.content
            
        except Exception as e:
            error_msg = f"步骤4-综合分析失败: {str(e)}"
            logger.error(f"[Workflow Error] {error_msg}")
            raise RuntimeError(error_msg) from e
    
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main(config_path: Optional[str] = None, verbose_level: Optional[int] = None):
    """主函数 - 演示基于LangChain和Workflow的反欺诈分析系统"""
    # 加载配置文件
    config = load_config(config_path)
    
    # 如果命令行指定了verbose级别，优先使用
    if verbose_level is not None:
        config.setdefault("workflow", {})["verbose_level"] = verbose_level
        setup_logging(verbose_level)
    
    # 检查API密钥配置
    azure_config = config.get("azure_ai_inference", {})
    api_key = azure_config.get("api_key")
    
    if not api_key:
        logger.error("错误: API密钥未配置")
        logger.error("请在以下方式中选择一种配置API密钥:")
        logger.error("1. 在config.json中设置azure_ai_inference.api_key")
        logger.error("2. 设置环境变量: set GITHUB_TOKEN=your_token_here")
        logger.error("获取令牌: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens")
        return
    
    # 创建LangChain工作流实例，传入配置
    workflow = LangChainFraudWorkflow(config)
    
    # 测试消息（与原代码相同）
    test_message = "【中国联通手机营业厅】疯狂大砍价，砍至0元免费拿：腾讯、优酷视频会员，充值折扣券等。升级至最新版本，移动、电信都可帮砍哦，立即抢u.10010.cn/qAdKD回复TD可退订。"
    
    try:
        logger.info("=== 基于LangChain和Workflow的反欺诈分析系统 ===")
        logger.info(f"分析消息: {test_message}")
        
        # 尝试连接MCP服务器（可选）
        # try:
        #     mcp_connected = await workflow.connect_mcp_server(
        #         "fetch-mcp",
        #         "node",
        #         ["C:\\src\\github\\fetch-mcp\\dist\\index.js"],
        #         {}
        #     )
        #     if mcp_connected:
        #         print("MCP服务器连接成功")
        #     else:
        #         print("MCP服务器连接失败，将使用内置工具")
        # except Exception as e:
        #     print(f"MCP服务器连接失败: {str(e)}，将使用内置工具")
        
        # 运行LangChain工作流
        logger.info("\n正在使用LangChain工作流进行分析...")
        result = await workflow.ainvoke({"text": test_message})
        
        # 输出结果
        print("\n" + "="*60)
        print("LangChain工作流分析结果:")
        print("="*60)
        
        if result.get("status") in ["failed", "model_failure", "error", "timeout"]:
            if result.get("status") == "timeout":
                print(f"\n工作流超时:")
                print(f"   错误详情: {result['error']}")
                if "stopped_at" in result:
                    print(f"   停止位置: {result['stopped_at']}")
                print(f"\n这可能是由于:")
                print(f"   1. 模型API响应过慢")
                print(f"   2. 网络连接不稳定")
                print(f"   3. 服务器负载过高")
            elif result.get("status") == "model_failure":
                print(f"\n模型调用失败，程序已停止:")
                print(f"   错误详情: {result['error']}")
                if "stopped_at" in result:
                    print(f"   停止位置: {result['stopped_at']}")
                print(f"\n请检查以下可能的原因:")
                print(f"   1. GITHUB_TOKEN 环境变量是否正确设置")
                print(f"   2. 网络连接是否正常")
                print(f"   3. Azure AI Inference 服务是否可用")
                print(f"   4. 模型 '{workflow.llm.model_name}' 是否可访问")
            elif result.get("status") == "failed":
                print(f"\n工作流在步骤{result.get('failed_step', '未知')}失败:")
                print(f"   错误详情: {result['error']}")
            elif result.get("status") == "partial_failure":
                print(f"\n工作流部分完成，步骤{result.get('failed_step', '未知')}失败:")
                print(f"   错误详情: {result['error']}")
                # 显示已完成的步骤
                print("\n已完成的步骤:")
                for step in result.get("workflow_steps", []):
                    if step.get("status") == "success":
                        print(f"  {step['step']}. {step['name']}: OK")
            else:
                print(f"\n程序执行失败: {result.get('error', '未知错误')}")
        else:
            # 显示工作流步骤
            print("\n工作流执行步骤:")
            for step in result.get("workflow_steps", []):
                status_icon = "✅" if step.get("status") == "success" else "❌"
                print(f"  {step['step']}. {step['name']}: {status_icon}")
            
            if result.get("final_analysis"):
                print(f"\n最终分析结果:")
                print(result["final_analysis"])
            else:
                print(f"\n最终分析未完成")
        
        # 保存完整结果到文件
        output_file = "langchain_fraud_analysis_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        if result.get("status") in ["failed", "model_failure", "error", "partial_failure", "timeout"]:
            logger.warning(f"\n错误信息已保存到: {output_file}")
        else:
            logger.info(f"\n完整分析结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"\n程序执行失败: {str(e)}")
        
    finally:
        # 清理资源
        await workflow.cleanup()
        logger.info("\n资源清理完成")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="反欺诈分析系统")
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        help="配置文件路径 (JSON格式)",
        default=None
    )
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        help="要分析的消息内容",
        default=None
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        choices=[0, 1, 2, 3],
        help="详细程度级别 (0=仅错误, 1=默认, 2=详细, 3=调试)",
        default=None
    )
    
    args = parser.parse_args()
    
    # 设置初始日志级别（如果命令行指定了）
    if args.verbose is not None:
        setup_logging(args.verbose)
    
    logger.info("启动严格基于LangChain和Workflow的反欺诈分析系统...")
    if args.config:
        logger.info(f"使用配置文件: {args.config}")
    if args.verbose is not None:
        logger.info(f"详细程度级别: {args.verbose}")
    
    asyncio.run(main(config_path=args.config, verbose_level=args.verbose))
