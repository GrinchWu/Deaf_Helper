"""
VLLM客户端 - 与阿里云DashScope API通信
使用Qwen3-Omni-flash模型进行视觉分析
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional, List, Dict, Any

import httpx

from .models import VLLMConfig

logger = logging.getLogger(__name__)


@dataclass
class VLLMRequest:
    """VLLM请求"""
    image_data: str  # Base64编码的图像
    prompt: str


@dataclass
class VLLMStreamChunk:
    """VLLM流式响应块"""
    content: str
    is_final: bool


class VLLMClientError(Exception):
    """VLLM客户端错误"""
    pass


class VLLMClient:
    """VLLM客户端 - 流式调用DashScope API"""

    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or VLLMConfig()
        self._client: Optional[httpx.AsyncClient] = None
        # 记录所有请求参数用于测试验证
        self._request_history: List[Dict[str, Any]] = []

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建HTTP客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def close(self) -> None:
        """关闭HTTP客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_request_body(self, request: VLLMRequest) -> Dict[str, Any]:
        """构建API请求体"""
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{request.image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": request.prompt
                        }
                    ]
                }
            ],
            "stream": True,  # 必须设置为True
        }
        return body


    async def analyze_stream(self, request: VLLMRequest) -> AsyncIterator[VLLMStreamChunk]:
        """
        流式调用VLLM API分析图像
        实现指数退避重试逻辑（最多3次）
        
        Args:
            request: VLLM请求
            
        Yields:
            VLLMStreamChunk: 流式响应块
        """
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        body = self._build_request_body(request)
        
        # 记录请求用于测试验证
        self._request_history.append({
            "url": url,
            "body": body,
            "stream": body.get("stream"),
        })
        
        last_error: Optional[Exception] = None
        retry_delays = [1.0, 2.0, 4.0]  # 指数退避延迟
        
        for attempt in range(self.config.max_retries):
            try:
                client = await self._get_client()
                
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status_code == 401:
                        raise VLLMClientError("Authentication failed: Invalid API key")
                    
                    if response.status_code == 429:
                        # Rate limit - 等待后重试
                        retry_after = float(response.headers.get("retry-after", retry_delays[attempt]))
                        logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status_code >= 500:
                        # 服务器错误 - 重试
                        raise httpx.HTTPStatusError(
                            f"Server error: {response.status_code}",
                            request=response.request,
                            response=response
                        )
                    
                    response.raise_for_status()
                    
                    # 处理流式响应
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                yield VLLMStreamChunk(content="", is_final=True)
                                return
                            try:
                                chunk_data = json.loads(data)
                                if "choices" in chunk_data and chunk_data["choices"]:
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield VLLMStreamChunk(content=content, is_final=False)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse chunk: {data}")
                                continue
                    
                    # 流结束
                    yield VLLMStreamChunk(content="", is_final=True)
                    return
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(f"Request error (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
        
        # 所有重试都失败
        raise VLLMClientError(f"All {self.config.max_retries} retries failed: {last_error}")

    async def analyze(self, request: VLLMRequest) -> str:
        """
        非流式调用 - 收集所有响应块并返回完整内容
        
        Args:
            request: VLLM请求
            
        Returns:
            完整的响应内容
        """
        content_parts = []
        async for chunk in self.analyze_stream(request):
            if not chunk.is_final:
                content_parts.append(chunk.content)
        return "".join(content_parts)

    def get_request_history(self) -> List[Dict[str, Any]]:
        """获取请求历史（用于测试）"""
        return self._request_history

    def clear_request_history(self) -> None:
        """清空请求历史"""
        self._request_history.clear()
