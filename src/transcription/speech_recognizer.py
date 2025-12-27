"""
语音识别器 - 调用大模型API进行语音转文字
支持流式识别
"""
import asyncio
import base64
import json
import logging
from typing import AsyncIterator, Optional, List

import httpx
import numpy as np

from .models import TranscriptionSegment, TranscriptionConfig
from .audio_processor import AudioChunk

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """语音识别器"""

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._buffer: List[AudioChunk] = []
        self._buffer_duration: float = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """关闭客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _audio_to_base64(self, audio_data: np.ndarray) -> str:
        """将音频数据转换为base64"""
        # 转换为16位整数
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return base64.b64encode(audio_int16.tobytes()).decode('utf-8')

    async def recognize_stream(
        self, 
        audio_chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptionSegment]:
        """
        流式语音识别
        
        Args:
            audio_chunks: 音频块流
            
        Yields:
            TranscriptionSegment: 转录片段
        """
        min_duration = 1.0  # 最小识别时长
        max_duration = 5.0  # 最大缓冲时长
        
        async for chunk in audio_chunks:
            # 只处理包含语音的块
            if not chunk.is_speech:
                # 如果缓冲区有内容且遇到静音，触发识别
                if self._buffer and self._buffer_duration >= min_duration:
                    result = await self._recognize_buffer()
                    if result:
                        yield result
                continue
            
            # 添加到缓冲区
            self._buffer.append(chunk)
            self._buffer_duration += chunk.end_time - chunk.start_time
            
            # 缓冲区达到最大时长，触发识别
            if self._buffer_duration >= max_duration:
                result = await self._recognize_buffer()
                if result:
                    yield result
        
        # 处理剩余缓冲区
        if self._buffer:
            result = await self._recognize_buffer()
            if result:
                yield result

    async def _recognize_buffer(self) -> Optional[TranscriptionSegment]:
        """识别缓冲区中的音频"""
        if not self._buffer:
            return None
        
        # 合并音频数据
        audio_data = np.concatenate([chunk.data for chunk in self._buffer])
        start_time = self._buffer[0].start_time
        end_time = self._buffer[-1].end_time
        
        # 清空缓冲区
        self._buffer.clear()
        self._buffer_duration = 0
        
        # 调用API识别
        try:
            text = await self._call_recognition_api(audio_data)
            if text:
                return TranscriptionSegment(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.9,
                )
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
        
        return None


    async def _call_recognition_api(self, audio_data: np.ndarray) -> Optional[str]:
        """
        调用语音识别API
        使用Qwen-Omni模型进行语音识别
        """
        client = await self._get_client()
        
        # 将音频编码为base64
        audio_base64 = self._audio_to_base64(audio_data)
        
        url = f"{self.config.api_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        # 构建请求
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/pcm;base64,{audio_base64}",
                                "format": "pcm"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请将这段音频转录为文字，只输出转录结果，不要添加任何解释。如果听不清或没有语音，输出[无语音]。"
                        }
                    ]
                }
            ],
            "stream": True,
        }
        
        try:
            text_parts = []
            async with client.stream("POST", url, headers=headers, json=body) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                text_parts.append(content)
                    except json.JSONDecodeError:
                        continue
            
            result = "".join(text_parts).strip()
            
            # 过滤无效结果
            if result in ["[无语音]", "", "无语音", "..."]:
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    async def recognize_single(self, audio_data: np.ndarray, start_time: float, end_time: float) -> Optional[TranscriptionSegment]:
        """
        识别单个音频片段
        """
        text = await self._call_recognition_api(audio_data)
        if text:
            return TranscriptionSegment(
                text=text,
                start_time=start_time,
                end_time=end_time,
                confidence=0.9,
            )
        return None
