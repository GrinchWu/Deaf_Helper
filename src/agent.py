"""
听障辅助Agent主类 - 整合所有组件
实现视频流处理、场景分析、服务决策的完整流程
采用事件驱动架构：只有场景变化时才调用API
"""
import asyncio
import base64
import logging
from typing import AsyncIterator, Optional

import cv2
import numpy as np

from .models import (
    AgentConfig, AgentOutput, ContextEntry,
    SceneAnalysisResult, ProcessorConfig, VLLMConfig, MemoryConfig,
    UserLocation, UserState
)
from .video_processor import VideoStreamProcessor
from .vllm_client import VLLMClient
from .scene_analyzer import SceneAnalyzer
from .service_decision import ServiceDecisionEngine
from .memory_manager import MemoryManager
from .scene_change_detector import SceneChangeDetector, ChangeDetectionConfig, ChangeType

logger = logging.getLogger(__name__)


class HearingImpairedAgent:
    """听障辅助Agent主类"""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        
        # 初始化各组件
        self.processor = VideoStreamProcessor(self.config.processor_config)
        self.vllm_client = VLLMClient(self.config.vllm_config)
        self.analyzer = SceneAnalyzer(self.vllm_client)
        self.memory = MemoryManager(self.config.memory_config)
        self.decision_engine = ServiceDecisionEngine(self.memory)
        
        # 场景变化检测器
        self.change_detector = SceneChangeDetector(ChangeDetectionConfig(
            frame_diff_threshold=0.05,
            scene_cut_threshold=0.3,
            histogram_threshold=0.7,
            min_interval_seconds=1.0,
            max_interval_seconds=10.0,
        ))
        
        # 统计信息
        self._total_frames = 0
        self._api_calls = 0
        self._last_analysis: Optional[SceneAnalysisResult] = None
        
        # 自适应采样相关
        self._consecutive_low_activity = 0
        self._base_sample_rate = self.config.processor_config.sample_rate

    async def run(self, video_source: str) -> AsyncIterator[AgentOutput]:
        """
        运行Agent处理视频源（事件驱动模式）
        只有当场景发生显著变化时才调用API
        
        Args:
            video_source: 视频文件路径或流地址
            
        Yields:
            AgentOutput: Agent输出结果
        """
        # 使用OpenCV直接读取视频以获取原始帧
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_source}")
        
        try:
            frame_id = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self._total_frames += 1
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                # 场景变化检测（本地快速检测）
                change_result = self.change_detector.detect_change(frame)
                
                # 只有需要调用API时才进行分析
                if change_result.should_call_api:
                    try:
                        # 编码帧为Base64
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        image_data = base64.b64encode(buffer).decode('utf-8')
                        
                        from .models import FrameData
                        frame_data = FrameData(
                            frame_id=frame_id,
                            timestamp=timestamp,
                            image_data=image_data,
                            width=frame.shape[1],
                            height=frame.shape[0],
                        )
                        
                        # 调用API分析
                        analysis = await self.analyzer.analyze_frame(frame_data)
                        self._api_calls += 1
                        self._last_analysis = analysis
                        
                        # 服务决策
                        decision = await self.decision_engine.decide(analysis)
                        
                        # 存储上下文
                        context_entry = ContextEntry(
                            frame_id=frame_id,
                            timestamp=timestamp,
                            analysis=analysis,
                            decision=decision,
                        )
                        self.memory.store_context(context_entry)
                        
                        # 输出结果
                        output = AgentOutput(
                            frame_id=frame_id,
                            timestamp=timestamp,
                            scene_analysis=analysis,
                            service_decision=decision,
                        )
                        
                        logger.info(
                            f"Frame {frame_id}: {change_result.reason} | "
                            f"API调用率: {self._api_calls}/{self._total_frames} "
                            f"({100*self._api_calls/self._total_frames:.1f}%)"
                        )
                        
                        yield output
                        
                    except Exception as e:
                        logger.error(f"Error analyzing frame {frame_id}: {e}")
                else:
                    # 场景稳定，使用上次的分析结果
                    if self._last_analysis and self._total_frames % 100 == 0:
                        logger.debug(
                            f"Frame {frame_id}: 场景稳定，跳过API调用 | "
                            f"API调用率: {self._api_calls}/{self._total_frames}"
                        )
                
                frame_id += 1
                
                # 控制处理速率（可选）
                await asyncio.sleep(0.001)  # 让出控制权
                    
        finally:
            cap.release()
            await self.vllm_client.close()


    async def _adjust_sampling_rate(self, analysis: SceneAnalysisResult) -> None:
        """
        根据场景活动度调整采样率
        低活动场景增加采样间隔以节省能耗
        """
        # 判断是否为低活动场景
        is_low_activity = self._is_low_activity(analysis)
        
        if is_low_activity:
            self._consecutive_low_activity += 1
            
            # 连续多帧低活动，增加采样间隔
            if self._consecutive_low_activity >= 5:
                new_rate = min(
                    self.processor.sample_rate + 1,
                    self._base_sample_rate * 3  # 最多增加到基础采样率的3倍
                )
                if new_rate != self.processor.sample_rate:
                    logger.debug(f"Increasing sample rate to {new_rate} due to low activity")
                    self.processor.sample_rate = new_rate
        else:
            # 高活动场景，恢复基础采样率
            self._consecutive_low_activity = 0
            if self.processor.sample_rate != self._base_sample_rate:
                logger.debug(f"Restoring sample rate to {self._base_sample_rate}")
                self.processor.sample_rate = self._base_sample_rate

    def _is_low_activity(self, analysis: SceneAnalysisResult) -> bool:
        """
        判断是否为低活动场景
        """
        # 获取最近的分析结果进行比较
        recent = self.memory.get_recent_context(3)
        
        if len(recent) < 2:
            return False
        
        # 比较场景变化
        prev_analysis = recent[-2].analysis
        
        # 位置和状态都没变化视为低活动
        location_same = analysis.user_location == prev_analysis.user_location
        state_same = analysis.user_state == prev_analysis.user_state
        
        # 物体变化不大
        prev_objects = set(prev_analysis.objects)
        curr_objects = set(analysis.objects)
        objects_similar = len(prev_objects.symmetric_difference(curr_objects)) <= 2
        
        return location_same and state_same and objects_similar

    async def run_fast(self, video_source: str) -> AsyncIterator[AgentOutput]:
        """
        快速运行模式（不控制速率，用于测试）
        """
        try:
            async for frame in self.processor.process_video_fast(video_source):
                try:
                    analysis = await self.analyzer.analyze_frame(frame)
                    decision = await self.decision_engine.decide(analysis)
                    
                    context_entry = ContextEntry(
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        analysis=analysis,
                        decision=decision,
                    )
                    self.memory.store_context(context_entry)
                    
                    output = AgentOutput(
                        frame_id=frame.frame_id,
                        timestamp=frame.timestamp,
                        scene_analysis=analysis,
                        service_decision=decision,
                    )
                    
                    yield output
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame.frame_id}: {e}")
                    continue
                    
        finally:
            await self.vllm_client.close()

    def get_statistics(self) -> dict:
        """获取运行统计信息"""
        api_call_rate = (self._api_calls / self._total_frames * 100) if self._total_frames > 0 else 0
        return {
            "total_frames": self._total_frames,
            "api_calls": self._api_calls,
            "api_call_rate": f"{api_call_rate:.1f}%",
            "frames_saved": self._total_frames - self._api_calls,
            "processed_contexts": self.memory.get_context_count(),
            "current_sample_rate": self.processor.sample_rate,
            "memory_usage_mb": self.memory.get_memory_usage_mb(),
            "change_detector_stats": self.change_detector.get_stats(),
        }

    async def close(self) -> None:
        """关闭Agent，释放资源"""
        await self.vllm_client.close()
