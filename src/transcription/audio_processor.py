"""
音频处理器 - 从视频提取音频并进行预处理
包括噪音去除、语音活动检测
"""
import asyncio
import logging
from typing import AsyncIterator, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import cv2

from .models import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """音频块"""
    data: np.ndarray           # 音频数据
    start_time: float          # 开始时间
    end_time: float            # 结束时间
    is_speech: bool            # 是否包含语音
    energy: float              # 能量值


class AudioProcessor:
    """音频处理器"""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._noise_profile: Optional[np.ndarray] = None

    def extract_audio_from_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        从视频文件提取音频
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            (音频数据, 采样率)
        """
        try:
            import subprocess
            import tempfile
            import os
            
            # 使用ffmpeg提取音频
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # 不要视频
                '-acodec', 'pcm_s16le',
                '-ar', str(self.config.sample_rate),
                '-ac', str(self.config.channels),
                '-y', tmp_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # 读取wav文件
            import wave
            with wave.open(tmp_path, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                audio_data /= 32768.0  # 归一化到 [-1, 1]
            
            os.unlink(tmp_path)
            return audio_data, self.config.sample_rate
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            # 返回空音频
            return np.array([], dtype=np.float32), self.config.sample_rate

    async def process_audio_stream(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> AsyncIterator[AudioChunk]:
        """
        流式处理音频数据
        
        Args:
            audio_data: 完整音频数据
            sample_rate: 采样率
            
        Yields:
            AudioChunk: 处理后的音频块
        """
        chunk_samples = int(self.config.chunk_duration * sample_rate)
        total_samples = len(audio_data)
        
        # 估计噪音profile（使用前几个chunk）
        if self.config.noise_reduce and self._noise_profile is None:
            noise_samples = min(chunk_samples * 3, total_samples)
            self._estimate_noise_profile(audio_data[:noise_samples])
        
        for i in range(0, total_samples, chunk_samples):
            chunk_data = audio_data[i:i + chunk_samples]
            
            if len(chunk_data) < chunk_samples // 2:
                continue
            
            start_time = i / sample_rate
            end_time = (i + len(chunk_data)) / sample_rate
            
            # 降噪处理
            if self.config.noise_reduce:
                chunk_data = self._reduce_noise(chunk_data)
            
            # 计算能量
            energy = np.sqrt(np.mean(chunk_data ** 2))
            
            # 语音活动检测
            is_speech = self._detect_speech(chunk_data, energy)
            
            yield AudioChunk(
                data=chunk_data,
                start_time=start_time,
                end_time=end_time,
                is_speech=is_speech,
                energy=energy,
            )
            
            # 让出控制权
            await asyncio.sleep(0.001)


    def _estimate_noise_profile(self, audio_data: np.ndarray) -> None:
        """估计噪音特征"""
        # 使用频谱分析估计噪音
        if len(audio_data) < 256:
            self._noise_profile = np.zeros(128)
            return
        
        # 简单的频谱估计
        fft = np.fft.rfft(audio_data[:256])
        self._noise_profile = np.abs(fft)

    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """
        简单的噪音去除
        使用谱减法的简化版本
        """
        if self._noise_profile is None or len(audio_data) < 256:
            return audio_data
        
        # 对于短音频，使用简单的阈值降噪
        threshold = self.config.noise_threshold
        
        # 软阈值处理
        result = audio_data.copy()
        mask = np.abs(result) < threshold
        result[mask] = 0
        
        return result

    def _detect_speech(self, audio_data: np.ndarray, energy: float) -> bool:
        """
        语音活动检测 (VAD)
        """
        if not self.config.vad_enabled:
            return True
        
        # 基于能量的简单VAD
        if energy < self.config.noise_threshold:
            return False
        
        # 计算过零率
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / 2
        zcr = zero_crossings / len(audio_data)
        
        # 语音通常有适中的过零率
        # 太高可能是噪音，太低可能是静音
        if zcr < 0.01 or zcr > 0.5:
            return False
        
        return energy > self.config.vad_threshold * self.config.noise_threshold

    def reset(self) -> None:
        """重置处理器状态"""
        self._noise_profile = None
