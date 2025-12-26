"""
场景变化检测器 - 本地轻量级检测
只有当场景发生显著变化时才触发API调用
"""
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

import cv2
import numpy as np


class ChangeType(Enum):
    """变化类型"""
    NO_CHANGE = "no_change"           # 无变化
    MINOR_CHANGE = "minor_change"     # 轻微变化
    SIGNIFICANT_CHANGE = "significant_change"  # 显著变化
    SCENE_CUT = "scene_cut"           # 场景切换


@dataclass
class ChangeDetectionConfig:
    """变化检测配置"""
    # 帧差阈值
    frame_diff_threshold: float = 0.05      # 5% 像素变化视为显著
    scene_cut_threshold: float = 0.3        # 30% 变化视为场景切换
    
    # 直方图比较阈值
    histogram_threshold: float = 0.7        # 相关性低于0.7视为变化
    
    # 时间控制
    min_interval_seconds: float = 1.0       # 最小API调用间隔
    max_interval_seconds: float = 10.0      # 最大API调用间隔（强制刷新）
    
    # 运动检测
    motion_threshold: float = 0.02          # 运动区域占比阈值
    
    # 稳定性检测
    stability_frames: int = 3               # 连续N帧稳定才认为场景稳定


@dataclass
class ChangeDetectionResult:
    """变化检测结果"""
    change_type: ChangeType
    should_call_api: bool
    frame_diff_score: float
    histogram_score: float
    motion_score: float
    reason: str


class SceneChangeDetector:
    """场景变化检测器"""

    def __init__(self, config: Optional[ChangeDetectionConfig] = None):
        self.config = config or ChangeDetectionConfig()
        
        # 状态
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._last_api_call_time: float = 0
        self._stable_frame_count: int = 0
        self._last_change_type: ChangeType = ChangeType.NO_CHANGE

    def detect_change(self, frame: np.ndarray) -> ChangeDetectionResult:
        """
        检测当前帧与上一帧的变化
        
        Args:
            frame: 当前帧 (BGR格式)
            
        Returns:
            ChangeDetectionResult: 检测结果
        """
        current_time = time.time()
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # 首帧处理
        if self._prev_frame is None:
            self._update_state(frame, gray, hist, current_time)
            return ChangeDetectionResult(
                change_type=ChangeType.SIGNIFICANT_CHANGE,
                should_call_api=True,
                frame_diff_score=1.0,
                histogram_score=0.0,
                motion_score=1.0,
                reason="首帧，需要初始化场景分析"
            )
        
        # 计算各项指标
        frame_diff_score = self._calculate_frame_diff(gray)
        histogram_score = self._calculate_histogram_similarity(hist)
        motion_score = self._calculate_motion_score(gray)
        
        # 判断变化类型
        change_type = self._determine_change_type(
            frame_diff_score, histogram_score, motion_score
        )
        
        # 判断是否需要调用API
        should_call_api, reason = self._should_call_api(
            change_type, current_time
        )
        
        # 更新状态
        if should_call_api:
            self._last_api_call_time = current_time
        
        self._update_state(frame, gray, hist, current_time)
        self._last_change_type = change_type
        
        return ChangeDetectionResult(
            change_type=change_type,
            should_call_api=should_call_api,
            frame_diff_score=frame_diff_score,
            histogram_score=histogram_score,
            motion_score=motion_score,
            reason=reason
        )


    def _calculate_frame_diff(self, gray: np.ndarray) -> float:
        """计算帧差分数 (0-1)"""
        if self._prev_gray is None:
            return 1.0
        
        # 计算绝对差
        diff = cv2.absdiff(gray, self._prev_gray)
        
        # 二值化
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # 计算变化像素占比
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        
        return changed_pixels / total_pixels

    def _calculate_histogram_similarity(self, hist: np.ndarray) -> float:
        """计算直方图相似度 (0-1, 1表示完全相同)"""
        if self._prev_hist is None:
            return 0.0
        
        # 使用相关性比较
        correlation = cv2.compareHist(
            self._prev_hist, hist, cv2.HISTCMP_CORREL
        )
        
        # 归一化到0-1
        return max(0.0, correlation)

    def _calculate_motion_score(self, gray: np.ndarray) -> float:
        """计算运动分数 (0-1)"""
        if self._prev_gray is None:
            return 1.0
        
        # 使用光流或简单差分
        diff = cv2.absdiff(gray, self._prev_gray)
        
        # 高斯模糊减少噪声
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # 二值化
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪点
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 计算运动区域占比
        motion_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        
        return motion_pixels / total_pixels

    def _determine_change_type(
        self, 
        frame_diff: float, 
        hist_sim: float, 
        motion: float
    ) -> ChangeType:
        """判断变化类型"""
        # 场景切换：帧差很大或直方图差异很大
        if frame_diff > self.config.scene_cut_threshold:
            return ChangeType.SCENE_CUT
        
        if hist_sim < 0.5:  # 直方图相关性很低
            return ChangeType.SCENE_CUT
        
        # 显著变化
        if frame_diff > self.config.frame_diff_threshold:
            return ChangeType.SIGNIFICANT_CHANGE
        
        if hist_sim < self.config.histogram_threshold:
            return ChangeType.SIGNIFICANT_CHANGE
        
        if motion > self.config.motion_threshold:
            return ChangeType.MINOR_CHANGE
        
        return ChangeType.NO_CHANGE

    def _should_call_api(
        self, 
        change_type: ChangeType, 
        current_time: float
    ) -> Tuple[bool, str]:
        """判断是否需要调用API"""
        time_since_last_call = current_time - self._last_api_call_time
        
        # 场景切换：立即调用
        if change_type == ChangeType.SCENE_CUT:
            return True, "检测到场景切换"
        
        # 显著变化：如果超过最小间隔，调用
        if change_type == ChangeType.SIGNIFICANT_CHANGE:
            if time_since_last_call >= self.config.min_interval_seconds:
                return True, "检测到显著场景变化"
            else:
                return False, f"场景变化但距上次调用仅{time_since_last_call:.1f}秒"
        
        # 超过最大间隔：强制刷新
        if time_since_last_call >= self.config.max_interval_seconds:
            return True, f"超过{self.config.max_interval_seconds}秒未更新，强制刷新"
        
        # 轻微变化或无变化：不调用
        return False, "场景稳定，无需调用API"

    def _update_state(
        self, 
        frame: np.ndarray, 
        gray: np.ndarray, 
        hist: np.ndarray,
        current_time: float
    ) -> None:
        """更新内部状态"""
        self._prev_frame = frame.copy()
        self._prev_gray = gray.copy()
        self._prev_hist = hist.copy()

    def reset(self) -> None:
        """重置检测器状态"""
        self._prev_frame = None
        self._prev_gray = None
        self._prev_hist = None
        self._last_api_call_time = 0
        self._stable_frame_count = 0
        self._last_change_type = ChangeType.NO_CHANGE

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "last_api_call_time": self._last_api_call_time,
            "last_change_type": self._last_change_type.value,
            "stable_frame_count": self._stable_frame_count,
        }
