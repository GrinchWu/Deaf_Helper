"""
转录Agent - 整合所有组件实现实时转录和情感分析
"""
import asyncio
import logging
import time
from typing import AsyncIterator, Optional, Tuple

import cv2
import numpy as np

from .models import (
    TranscriptionConfig, TranscriptionResult, TranscriptionSegment,
    EmotionResult
)
from .audio_processor import AudioProcessor, AudioChunk
from .speech_recognizer import SpeechRecognizer
from .emotion_analyzer import EmotionAnalyzer
from .subtitle_renderer import SubtitleRenderer

logger = logging.getLogger(__name__)


class TranscriptionAgent:
    """实时转录与情感分析Agent"""

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        
        # 初始化组件
        self.audio_processor = AudioProcessor(self.config.audio_config)
        self.speech_recognizer = SpeechRecognizer(self.config)
        self.emotion_analyzer = EmotionAnalyzer(self.config)
        self.subtitle_renderer = SubtitleRenderer(self.config.subtitle_style)
        
        # 状态
        self._current_result: Optional[TranscriptionResult] = None
        self._last_emotion_update: float = 0
        
        # 统计
        self._total_segments = 0
        self._total_duration = 0.0

    async def process_video(
        self, 
        video_path: str
    ) -> AsyncIterator[Tuple[np.ndarray, Optional[TranscriptionResult]]]:
        """
        处理视频，返回带字幕的帧流
        
        Args:
            video_path: 视频文件路径
            
        Yields:
            (带字幕的帧, 转录结果)
        """
        # 提取音频
        logger.info(f"Extracting audio from {video_path}")
        audio_data, sample_rate = self.audio_processor.extract_audio_from_video(video_path)
        
        if len(audio_data) == 0:
            logger.warning("No audio extracted, processing video without transcription")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {total_frames} frames at {fps} fps")
        
        try:
            # 启动音频处理任务
            transcription_task = asyncio.create_task(
                self._process_audio(audio_data, sample_rate)
            )
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_idx / fps
                
                # 渲染字幕
                rendered_frame = self.subtitle_renderer.render(
                    frame,
                    self._current_result,
                    current_time,
                    show_emotion=self.config.show_emotion_indicator,
                    show_intent=self.config.show_speaker_intent,
                )
                
                yield rendered_frame, self._current_result
                
                frame_idx += 1
                
                # 控制处理速率
                await asyncio.sleep(0.001)
            
            # 等待转录完成
            await transcription_task
            
        finally:
            cap.release()
            await self.close()


    async def _process_audio(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """处理音频流"""
        if len(audio_data) == 0:
            return
        
        # 处理音频块
        async for chunk in self.audio_processor.process_audio_stream(audio_data, sample_rate):
            if not chunk.is_speech:
                continue
            
            # 识别语音
            segment = await self.speech_recognizer.recognize_single(
                chunk.data, chunk.start_time, chunk.end_time
            )
            
            if segment and segment.text:
                # 情感分析
                emotion = None
                current_time = time.time()
                
                if (self.config.enable_emotion and 
                    current_time - self._last_emotion_update >= self.config.emotion_update_interval):
                    emotion = await self.emotion_analyzer.analyze(segment)
                    self._last_emotion_update = current_time
                
                # 更新当前结果
                self._current_result = TranscriptionResult(
                    segment=segment,
                    emotion=emotion,
                    timestamp=current_time,
                )
                
                self._total_segments += 1
                self._total_duration += segment.end_time - segment.start_time
                
                logger.info(
                    f"Transcribed: [{segment.start_time:.1f}s-{segment.end_time:.1f}s] "
                    f"{segment.text}"
                )
                
                if emotion:
                    logger.info(
                        f"  Emotion: {emotion.emotion.value} "
                        f"(intent: {emotion.speaker_intent})"
                    )

    async def process_realtime(
        self,
        video_source: str,
        output_path: Optional[str] = None,
        display: bool = True
    ) -> None:
        """
        实时处理视频并显示/保存结果
        
        Args:
            video_source: 视频源（文件路径或摄像头索引）
            output_path: 输出视频路径（可选）
            display: 是否显示窗口
        """
        writer = None
        
        try:
            async for frame, result in self.process_video(video_source):
                # 初始化视频写入器
                if output_path and writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
                
                # 写入帧
                if writer:
                    writer.write(frame)
                
                # 显示
                if display:
                    cv2.imshow('Transcription', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        finally:
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "total_segments": self._total_segments,
            "total_duration": f"{self._total_duration:.1f}s",
            "current_result": self._current_result.segment.text if self._current_result else None,
        }

    async def close(self) -> None:
        """关闭资源"""
        await self.speech_recognizer.close()
        await self.emotion_analyzer.close()

    def reset(self) -> None:
        """重置状态"""
        self._current_result = None
        self._last_emotion_update = 0
        self._total_segments = 0
        self._total_duration = 0.0
        self.audio_processor.reset()
        self.emotion_analyzer.clear_context()
        self.subtitle_renderer.clear()
