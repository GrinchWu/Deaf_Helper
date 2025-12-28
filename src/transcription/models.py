"""转录模块数据模型"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class Emotion(Enum):
    """情感类型"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"


@dataclass
class TranscriptSegment:
    """转录片段"""
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None


@dataclass
class EmotionResult:
    """情感分析结果"""
    emotion: Emotion
    confidence: float
    speaker_intent: str      # 说话者意图
    mental_state: str        # 心理状态推断
    suggested_response: str  # 建议回应


@dataclass
class TranscriptionResult:
    """完整转录结果"""
    segment: TranscriptSegment
    emotion: Optional[EmotionResult] = None
