"""
视频流处理器 - 逐帧处理视频文件
支持采样率配置和损坏帧跳过
"""
import asyncio
import base64
import logging
import time
from typing import AsyncIterator, Optional

import cv2
import numpy as np

from .models import FrameData, ProcessorConfig

logger = logging.getLogger(__name__)


class VideoProcessorError(Exception):
    """视频处理器错误"""
    pass


class VideoStreamProcessor:
    """视频流处理器"""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self._current_sample_rate = self.config.sample_rate

    @property
    def sample_rate(self) -> int:
        """当前采样率"""
        return self._current_sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        """设置采样率"""
        self._current_sample_rate = max(1, value)

    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """将帧编码为Base64字符串"""
        # 编码为JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise VideoProcessorError("Failed to encode frame to JPEG")
        
        # 转换为Base64
        return base64.b64encode(buffer).decode('utf-8')

    async def process_video(self, video_path: str) -> AsyncIterator[FrameData]:
        """
        将视频文件作为流处理，异步生成帧数据
        
        Args:
            video_path: 视频文件路径
            
        Yields:
            FrameData: 提取的帧数据
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise VideoProcessorError(f"Cannot open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0 / min(fps, self.config.max_fps)
            
            frame_count = 0
            yielded_count = 0
            last_frame_time = time.time()
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    # 视频结束或读取失败
                    break
                
                frame_count += 1
                
                # 采样逻辑：每sample_rate帧处理一次
                if (frame_count - 1) % self._current_sample_rate != 0:
                    continue
                
                try:
                    # 编码帧
                    image_data = self._encode_frame_to_base64(frame)
                    height, width = frame.shape[:2]
                    
                    # 计算时间戳
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    frame_data = FrameData(
                        frame_id=yielded_count,
                        timestamp=timestamp,
                        image_data=image_data,
                        width=width,
                        height=height,
                    )
                    
                    yielded_count += 1
                    yield frame_data
                    
                    # 控制处理速率
                    elapsed = time.time() - last_frame_time
                    if elapsed < frame_interval:
                        await asyncio.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()
                    
                except Exception as e:
                    # 损坏帧跳过，继续处理
                    logger.warning(f"Skipping corrupted frame {frame_count}: {e}")
                    continue
                    
        finally:
            cap.release()


    async def process_video_fast(self, video_path: str) -> AsyncIterator[FrameData]:
        """
        快速处理视频（不控制速率，用于测试）
        
        Args:
            video_path: 视频文件路径
            
        Yields:
            FrameData: 提取的帧数据
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise VideoProcessorError(f"Cannot open video file: {video_path}")
        
        try:
            frame_count = 0
            yielded_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # 采样逻辑
                if (frame_count - 1) % self._current_sample_rate != 0:
                    continue
                
                try:
                    image_data = self._encode_frame_to_base64(frame)
                    height, width = frame.shape[:2]
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    frame_data = FrameData(
                        frame_id=yielded_count,
                        timestamp=timestamp,
                        image_data=image_data,
                        width=width,
                        height=height,
                    )
                    
                    yielded_count += 1
                    yield frame_data
                    
                except Exception as e:
                    logger.warning(f"Skipping corrupted frame {frame_count}: {e}")
                    continue
                    
        finally:
            cap.release()

    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise VideoProcessorError(f"Cannot open video file: {video_path}")
        
        try:
            info = {
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                    if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
            }
            return info
        finally:
            cap.release()

    def count_frames_with_sampling(self, total_frames: int, sample_rate: int) -> int:
        """
        计算采样后的帧数
        
        Args:
            total_frames: 总帧数
            sample_rate: 采样率
            
        Returns:
            采样后的帧数
        """
        if total_frames <= 0 or sample_rate <= 0:
            return 0
        return (total_frames + sample_rate - 1) // sample_rate
