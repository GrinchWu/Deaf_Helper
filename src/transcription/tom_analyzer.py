"""
Theory of Mind (ToM) 情感分析器
基于 SimToM (Simulation Theory of Mind) 方法改进
参考论文: "Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities"

核心改进:
1. 两阶段推理 (Two-stage prompting)
   - Stage 1: 视角过滤 - 识别说话者所知道的信息
   - Stage 2: 心理状态推断 - 基于说话者视角推断情感和意图
2. BDI模型 (Belief-Desire-Intention) 
   - 信念: 说话者相信什么是真的
   - 欲望: 说话者想要什么
   - 意图: 说话者打算做什么
"""
import json
from typing import Optional
from dataclasses import dataclass, field

import httpx

try:
    from .models import Emotion, EmotionResult, TranscriptSegment
except ImportError:
    from models import Emotion, EmotionResult, TranscriptSegment


@dataclass
class ToMConfig:
    """ToM分析器配置"""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model: str = "qwen-plus"
    timeout: float = 30.0


@dataclass 
class BDIState:
    """BDI心理状态模型"""
    beliefs: list[str] = field(default_factory=list)      # 说话者的信念
    desires: list[str] = field(default_factory=list)      # 说话者的欲望
    intentions: list[str] = field(default_factory=list)   # 说话者的意图


class SimToMAnalyzer:
    """
    基于SimToM的两阶段心智理论分析器
    
    Stage 1: Perspective Filtering (视角过滤)
    - 识别说话者在说话时所知道/相信的信息
    - 过滤掉说话者不可能知道的信息
    
    Stage 2: Mental State Inference (心理状态推断)  
    - 基于过滤后的信息，从说话者视角推断心理状态
    - 使用BDI模型分析信念、欲望、意图
    """

    def __init__(self, config: Optional[ToMConfig] = None):
        self.config = config or ToMConfig()
        self._client: Optional[httpx.AsyncClient] = None

    def _build_stage1_prompt(self, text: str, context: str = "") -> str:
        """
        Stage 1: 视角过滤 Prompt
        目标: 识别说话者所知道的信息，建立说话者的知识边界
        """
        prompt = f"""你是一个心智理论(Theory of Mind)专家。请进行视角过滤分析。

## 任务
分析说话者在说出这句话时，他/她可能知道或相信的信息。
注意：只关注说话者的视角，不要假设说话者知道他们不可能知道的事情。

## 对话上下文
{context if context else "（无上下文）"}

## 当前发言
"{text}"

## 请分析并输出JSON
{{
  "speaker_knowledge": [
    "说话者知道/相信的事实1",
    "说话者知道/相信的事实2"
  ],
  "speaker_uncertainty": [
    "说话者可能不确定的事情1"
  ],
  "emotional_cues": [
    "从语言中识别到的情感线索1"
  ]
}}

只输出JSON，不要其他内容。"""
        return prompt

    def _build_stage2_prompt(self, text: str, stage1_result: dict, context: str = "") -> str:
        """
        Stage 2: 心理状态推断 Prompt
        基于Stage1的视角过滤结果，进行深度心理状态分析
        使用BDI模型 + 情感分析
        """
        knowledge = stage1_result.get("speaker_knowledge", [])
        uncertainty = stage1_result.get("speaker_uncertainty", [])
        emotional_cues = stage1_result.get("emotional_cues", [])
        
        prompt = f"""你是一个心智理论(Theory of Mind)专家。请基于视角过滤结果进行心理状态推断。

## 核心原则：模拟理论 (Simulation Theory)
想象你就是说话者本人。基于说话者所知道的信息，推断他们的心理状态。

## 说话者的知识边界（来自Stage 1分析）
- 说话者知道/相信: {json.dumps(knowledge, ensure_ascii=False)}
- 说话者不确定: {json.dumps(uncertainty, ensure_ascii=False)}  
- 情感线索: {json.dumps(emotional_cues, ensure_ascii=False)}

## 对话上下文
{context if context else "（无上下文）"}

## 当前发言
"{text}"

## 请使用BDI模型分析并输出JSON
{{
  "emotion": "情感类型(happy/sad/angry/fearful/surprised/disgusted/neutral)",
  "confidence": 0.0-1.0的置信度,
  "bdi": {{
    "beliefs": ["说话者相信什么是真的"],
    "desires": ["说话者想要什么"],
    "intentions": ["说话者打算做什么/希望达成什么"]
  }},
  "speaker_intent": "说话者说这句话的真实意图（一句话总结）",
  "mental_state": "说话者当前的心理状态描述",
  "perspective_insight": "从说话者视角看，他们为什么会这样说",
  "suggested_response": "建议听者如何回应（对听障用户的提示）"
}}

只输出JSON，不要其他内容。"""
        return prompt

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM并返回响应内容"""
        client = await self._get_client()
        
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
        
        return content

    def _extract_json(self, content: str) -> dict:
        """从LLM响应中提取JSON"""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except:
            pass
        return {}

    async def analyze(
        self, 
        segment: TranscriptSegment, 
        context: str = ""
    ) -> EmotionResult:
        """
        两阶段ToM分析
        
        Args:
            segment: 转录片段
            context: 对话上下文
            
        Returns:
            EmotionResult: 情感分析结果
        """
        text = segment.text
        
        # Stage 1: 视角过滤
        stage1_prompt = self._build_stage1_prompt(text, context)
        stage1_response = await self._call_llm(stage1_prompt)
        stage1_result = self._extract_json(stage1_response)
        
        # Stage 2: 心理状态推断
        stage2_prompt = self._build_stage2_prompt(text, stage1_result, context)
        stage2_response = await self._call_llm(stage2_prompt)
        stage2_result = self._extract_json(stage2_response)
        
        # 解析最终结果
        return self._parse_result(stage2_result)

    def _parse_result(self, data: dict) -> EmotionResult:
        """解析Stage 2的结果"""
        if not data:
            return self._default_result()
            
        try:
            emotion_str = data.get("emotion", "neutral").lower()
            emotion = Emotion.NEUTRAL
            for e in Emotion:
                if e.value == emotion_str:
                    emotion = e
                    break
            
            # 构建意图描述（包含BDI信息）
            bdi = data.get("bdi", {})
            intent_parts = []
            if data.get("speaker_intent"):
                intent_parts.append(data["speaker_intent"])
            if bdi.get("desires"):
                intent_parts.append(f"想要: {', '.join(bdi['desires'][:2])}")
            
            speaker_intent = " | ".join(intent_parts) if intent_parts else "未知"
            
            # 构建心理状态描述（包含视角洞察）
            mental_parts = []
            if data.get("mental_state"):
                mental_parts.append(data["mental_state"])
            if data.get("perspective_insight"):
                mental_parts.append(f"[视角] {data['perspective_insight']}")
                
            mental_state = " ".join(mental_parts) if mental_parts else "未知"
            
            return EmotionResult(
                emotion=emotion,
                confidence=float(data.get("confidence", 0.5)),
                speaker_intent=speaker_intent,
                mental_state=mental_state,
                suggested_response=data.get("suggested_response", ""),
            )
        except:
            return self._default_result()
    
    def _default_result(self) -> EmotionResult:
        """默认结果"""
        return EmotionResult(
            emotion=Emotion.NEUTRAL,
            confidence=0.3,
            speaker_intent="无法推断",
            mental_state="无法推断",
            suggested_response="",
        )

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# 保持向后兼容
ToMAnalyzer = SimToMAnalyzer
