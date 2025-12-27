"""
实时语音转录与情感分析模块
- 实时语音识别（流式处理）
- 环境噪音去除
- 字幕叠加到视频帧
- 基于ToM的情感推断
"""

from .audio_processor import AudioProcessor
from .speech_recognizer import SpeechRecognizer
from .emotion_analyzer import EmotionAnalyzer
from .subtitle_renderer import SubtitleRenderer
from .transcription_agent import TranscriptionAgent

__all__ = [
    "AudioProcessor",
    "SpeechRecognizer", 
    "EmotionAnalyzer",
    "SubtitleRenderer",
    "TranscriptionAgent",
]
