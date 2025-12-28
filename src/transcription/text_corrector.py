"""
异步文本修正器
结合上下文语境，对语音识别的文本进行修正
不阻塞主转录流程
"""
import json
from typing import Optional, Callable
from dataclasses import dataclass

import httpx


@dataclass
class CorrectorConfig:
    """文本修正器配置"""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model: str = "qwen-flash"
    timeout: float = 15.0


class TextCorrector:
    """
    异步文本修正器
    
    功能:
    1. 修正语音识别中的同音字错误
    2. 修正断句错误
    3. 补全不完整的句子
    4. 根据上下文修正专业术语
    """

    def __init__(self, config: Optional[CorrectorConfig] = None):
        self.config = config or CorrectorConfig()
        self._client: Optional[httpx.AsyncClient] = None

    def _build_correction_prompt(self, text: str, context: list[str]) -> str:
        """构建文本修正 Prompt"""
        context_str = "\n".join([f"- {c}" for c in context[-5:]]) if context else "（无上下文）"
        
        prompt = f"""你是一个语音识别文本修正专家。只修正影响理解的严重错误，保留口语化表达。

## 核心原则：只修正影响理解的错误

### ✅ 需要修正的情况（影响理解）

1. **同音字导致意思完全错误**
   - "我想吃苹果" 误识别为 "我想吃平国" → 修正为 "苹果"
   - "今天开会" 误识别为 "今天开灰" → 修正为 "开会"
   - "手机没电了" 误识别为 "手机没店了" → 修正为 "没电"

2. **关键词识别错误导致语义不通**
   - "我在北京" 误识别为 "我在背景" → 修正为 "北京"
   - "这个项目" 误识别为 "这个香木" → 修正为 "项目"

3. **人名/地名/专业术语明显错误**
   - 上下文提到"张三"，后面出现"长三" → 修正为 "张三"
   - 讨论技术时，"API接口" 误识别为 "挨劈爱接口" → 修正

4. **断句导致意思完全相反或混乱**
   - "不要，停下来" 误识别为 "不要停，下来" → 需要修正

### ❌ 不需要修正的情况（保持原样）

1. **口语化表达、语气词**
   - "嗯"、"啊"、"那个"、"就是说" → 保持原样
   - "然后呢"、"对对对"、"好的好的" → 保持原样

2. **不规范但能理解的表达**
   - "我觉得这个挺好的吧" → 保持原样（虽然口语化）
   - "他那个人吧，怎么说呢" → 保持原样

3. **的/地/得 用法不规范**
   - "他跑的很快" → 保持原样（虽然应该是"得"，但不影响理解）
   - "我高兴的跳起来" → 保持原样

4. **轻微的语法问题**
   - "我和他一起去的" → 保持原样
   - "这个东西很好用的" → 保持原样

5. **重复、停顿、自我纠正**
   - "我我我想说" → 保持原样
   - "就是那个，那个什么" → 保持原样

## 上下文（之前的对话）
{context_str}

## 需要修正的文本
"{text}"

## 请输出JSON
{{
  "corrected_text": "修正后的文本（如无需修正则与原文相同）",
  "corrections": [
    {{"original": "原词", "corrected": "修正词", "reason": "修正原因"}}
  ],
  "confidence": 0.0-1.0的修正置信度,
  "needs_correction": true或false
}}

重要：大多数情况下不需要修正！只有严重影响理解时才修正。
只输出JSON，不要其他内容。"""
        return prompt

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def correct(self, text: str, context: list[str]) -> dict:
        """
        修正文本
        
        Args:
            text: 需要修正的文本
            context: 上下文列表（之前的句子）
            
        Returns:
            dict: {
                "original": 原文,
                "corrected": 修正后文本,
                "corrections": 修正列表,
                "confidence": 置信度
            }
        """
        prompt = self._build_correction_prompt(text, context)
        
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
            )
            
            content = ""
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content += delta.get("content", "")
                    except:
                        continue
            
            return self._parse_result(text, content)
            
        except Exception as e:
            return {
                "original": text,
                "corrected": text,
                "corrections": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def _parse_result(self, original: str, content: str) -> dict:
        """解析LLM返回结果"""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return {
                    "original": original,
                    "corrected": data.get("corrected_text", original),
                    "corrections": data.get("corrections", []),
                    "confidence": float(data.get("confidence", 0.5))
                }
        except:
            pass
        
        return {
            "original": original,
            "corrected": original,
            "corrections": [],
            "confidence": 0.0
        }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
