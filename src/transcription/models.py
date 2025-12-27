"""
转录模块数据模型
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class EmotionType(Enum):
    """情感类型"""
    HAPPY = "happy"           # 开心
    SAD = "sad"               # 悲伤
    ANGRY = "angry"           # 愤怒
    FEARFUL = "fearful"       # 恐惧
    SURPRISED = "surprised"   # 惊讶
    DISGUSTED = "disgusted"   # 厌恶
    NEUTRAL = "neutral"       # 中性
    CONFUSED = "confused"     # 困惑
    EXCITED = "excited"       # 兴奋
    ANXIOUS = "anxious"       # 焦虑


@dataclass
class TranscriptionSegment:
    """转录片段"""
    text: str                          # 转录文本
    start_time: float                  # 开始时间(秒)
    end_time: float                    # 结束时间(秒)
    confidence: float                  # 置信度
    speaker_id: Optional[str] = None   # 说话人ID


@dataclass
class EmotionResult:
    """情感分析结果"""
    emotion: EmotionType               # 主要情感
    confidence: float                  # 置信度
    intensity: float                   # 强度 (0-1)
    valence: float                     # 效价 (-1到1, 负面到正面)
    arousal: float                     # 唤醒度 (0-1, 平静到激动)
    
    # ToM推断结果
    speaker_intent: str                # 说话者意图
    speaker_belief: str                # 说话者信念
    speaker_desire: str                # 说话者期望
    mental_state_summary: str          # 心理状态总结


@dataclass
class TranscriptionResult:
    """完整转录结果"""
    segment: TranscriptionSegment      # 转录片段
    emotion: Optional[EmotionResult]   # 情感分析
    timestamp: float                   # 处理时间戳


@dataclass
class SubtitleStyle:
    """字幕样式"""
    font_name: str = "SimHei"          # 字体
    font_size: int = 32                # 字号
    font_color: tuple = (255, 255, 255)  # 白色
    bg_color: tuple = (0, 0, 0)        # 黑色背景
    bg_opacity: float = 0.7            # 背景透明度
    position: str = "bottom"           # 位置: bottom, top, center
    margin: int = 20                   # 边距
    
    # 情感颜色映射
    emotion_colors: Dict[EmotionType, tuple] = field(default_factory=lambda: {
        EmotionType.HAPPY: (0, 255, 100),      # 绿色
        EmotionType.SAD: (100, 100, 255),      # 蓝色
        EmotionType.ANGRY: (0, 0, 255),        # 红色
        EmotionType.FEARFUL: (128, 0, 128),    # 紫色
        EmotionType.SURPRISED: (0, 255, 255),  # 黄色
        EmotionType.NEUTRAL: (255, 255, 255),  # 白色
        EmotionType.EXCITED: (0, 165, 255),    # 橙色
        EmotionType.ANXIOUS: (180, 180, 180),  # 灰色
    })


@dataclass
class AudioConfig:
    """音频处理配置"""
    sample_rate: int = 16000           # 采样率
    channels: int = 1                  # 声道数
    chunk_duration: float = 0.5        # 音频块时长(秒)
    noise_reduce: bool = True          # 是否降噪
    noise_threshold: float = 0.02      # 噪音阈值
    vad_enabled: bool = True           # 语音活动检测
    vad_threshold: float = 0.5         # VAD阈值


@dataclass
class TranscriptionConfig:
    """转录配置"""
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    subtitle_style: SubtitleStyle = field(default_factory=SubtitleStyle)
    
    # API配置
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model: str = "qwen-omni-turbo"
    
    # 情感分析配置
    enable_emotion: bool = True
    tom_analysis: bool = True          # 启用ToM分析
    emotion_update_interval: float = 2.0  # 情感更新间隔(秒)
    
    # 显示配置
    show_emotion_indicator: bool = True
    show_speaker_intent: bool = True
