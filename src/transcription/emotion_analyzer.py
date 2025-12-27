"""
情感分析器 - 基于Theory of Mind (ToM)的情感推断
使用prompt指令提升模型的心智理论能力
"""
import asyncio
import json
import logging
from typing import Optional, List

import httpx

from .models import (
    TranscriptionSegment, EmotionResult, EmotionType, 
    TranscriptionConfig
)

logger = logging.getLogger(__name__)


# ToM增强的情感分析Prompt
TOM_EMOTION_PROMPT = """你是一个具有高度心智理论(Theory of Mind)能力的情感分析专家。
请分析以下对话文本，推断说话者的情感状态和心理活动。

## 分析步骤（请按步骤思考）：

### 第一步：观察者视角
从第三人称视角观察这段话，注意：
- 说话者使用的词汇和语气
- 表达的内容和方式
- 可能的语境线索

### 第二步：说话者视角（心智理论核心）
设身处地站在说话者的角度思考：
- 他/她说这句话时可能在想什么？（信念 Belief）
- 他/她想要达成什么目的？（意图 Intent）
- 他/她期望听者如何反应？（期望 Desire）
- 他/她当时的情绪状态是什么？（情感 Emotion）

### 第三步：情感推断
基于以上分析，判断说话者的：
- 主要情感类型
- 情感强度
- 情感效价（正面/负面）
- 唤醒程度（平静/激动）

## 待分析文本：
"{text}"

## 上下文（如有）：
{context}

## 请以JSON格式输出分析结果：
```json
{{
  "emotion": "情感类型(happy/sad/angry/fearful/surprised/disgusted/neutral/confused/excited/anxious)",
  "confidence": 0.0-1.0,
  "intensity": 0.0-1.0,
  "valence": -1.0到1.0,
  "arousal": 0.0-1.0,
  "speaker_intent": "说话者的意图（一句话描述）",
  "speaker_belief": "说话者的信念/想法（一句话描述）",
  "speaker_desire": "说话者的期望（一句话描述）",
  "mental_state_summary": "心理状态总结（一句话）"
}}
```

只输出JSON，不要其他内容。"""


class EmotionAnalyzer:
    """基于ToM的情感分析器"""

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._context_buffer: List[str] = []  # 上下文缓冲
        self._max_context: int = 5  # 最大上下文条数

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

    async def analyze(self, segment: TranscriptionSegment) -> Optional[EmotionResult]:
        """
        分析转录片段的情感
        
        Args:
            segment: 转录片段
            
        Returns:
            EmotionResult: 情感分析结果
        """
        if not self.config.enable_emotion:
            return None
        
        if not segment.text or len(segment.text.strip()) < 2:
            return None
        
        try:
            # 构建上下文
            context = self._build_context()
            
            # 调用API进行ToM分析
            result = await self._call_tom_analysis(segment.text, context)
            
            # 更新上下文
            self._update_context(segment.text)
            
            return result
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return self._default_emotion()

    def _build_context(self) -> str:
        """构建对话上下文"""
        if not self._context_buffer:
            return "无上下文"
        return "\n".join([f"- {text}" for text in self._context_buffer[-self._max_context:]])

    def _update_context(self, text: str) -> None:
        """更新上下文缓冲"""
        self._context_buffer.append(text)
        if len(self._context_buffer) > self._max_context * 2:
            self._context_buffer = self._context_buffer[-self._max_context:]


    async def _call_tom_analysis(self, text: str, context: str) -> Optional[EmotionResult]:
        """
        调用API进行ToM情感分析
        """
        client = await self._get_client()
        
        # 构建prompt
        prompt = TOM_EMOTION_PROMPT.format(text=text, context=context)
        
        url = f"{self.config.api_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个情感分析专家，擅长运用心智理论(Theory of Mind)推断他人的情感和心理状态。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True,
        }
        
        try:
            response_parts = []
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
                                response_parts.append(content)
                    except json.JSONDecodeError:
                        continue
            
            response_text = "".join(response_parts)
            return self._parse_emotion_response(response_text)
            
        except Exception as e:
            logger.error(f"ToM analysis API call failed: {e}")
            return None

    def _parse_emotion_response(self, response: str) -> Optional[EmotionResult]:
        """解析API响应"""
        try:
            # 提取JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                return self._default_emotion()
            
            json_str = response[start:end]
            data = json.loads(json_str)
            
            # 解析情感类型
            emotion_str = data.get("emotion", "neutral").lower()
            try:
                emotion = EmotionType(emotion_str)
            except ValueError:
                emotion = EmotionType.NEUTRAL
            
            return EmotionResult(
                emotion=emotion,
                confidence=float(data.get("confidence", 0.7)),
                intensity=float(data.get("intensity", 0.5)),
                valence=float(data.get("valence", 0.0)),
                arousal=float(data.get("arousal", 0.5)),
                speaker_intent=data.get("speaker_intent", "表达想法"),
                speaker_belief=data.get("speaker_belief", "未知"),
                speaker_desire=data.get("speaker_desire", "被理解"),
                mental_state_summary=data.get("mental_state_summary", "正常状态"),
            )
            
        except Exception as e:
            logger.error(f"Failed to parse emotion response: {e}")
            return self._default_emotion()

    def _default_emotion(self) -> EmotionResult:
        """返回默认情感结果"""
        return EmotionResult(
            emotion=EmotionType.NEUTRAL,
            confidence=0.5,
            intensity=0.3,
            valence=0.0,
            arousal=0.3,
            speaker_intent="表达信息",
            speaker_belief="一般陈述",
            speaker_desire="传达信息",
            mental_state_summary="平静状态",
        )

    def clear_context(self) -> None:
        """清空上下文"""
        self._context_buffer.clear()
