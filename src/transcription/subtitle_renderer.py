"""
字幕渲染器 - 将转录文字和情感信息叠加到视频帧上
"""
import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np

from .models import (
    TranscriptionResult, SubtitleStyle, EmotionType, EmotionResult
)

logger = logging.getLogger(__name__)


# 情感表情符号映射
EMOTION_EMOJI = {
    EmotionType.HAPPY: "😊",
    EmotionType.SAD: "😢",
    EmotionType.ANGRY: "😠",
    EmotionType.FEARFUL: "😨",
    EmotionType.SURPRISED: "😲",
    EmotionType.DISGUSTED: "🤢",
    EmotionType.NEUTRAL: "😐",
    EmotionType.CONFUSED: "😕",
    EmotionType.EXCITED: "🤩",
    EmotionType.ANXIOUS: "😰",
}

# 情感中文名称
EMOTION_NAMES = {
    EmotionType.HAPPY: "开心",
    EmotionType.SAD: "悲伤",
    EmotionType.ANGRY: "愤怒",
    EmotionType.FEARFUL: "恐惧",
    EmotionType.SURPRISED: "惊讶",
    EmotionType.DISGUSTED: "厌恶",
    EmotionType.NEUTRAL: "平静",
    EmotionType.CONFUSED: "困惑",
    EmotionType.EXCITED: "兴奋",
    EmotionType.ANXIOUS: "焦虑",
}


class SubtitleRenderer:
    """字幕渲染器"""

    def __init__(self, style: Optional[SubtitleStyle] = None):
        self.style = style or SubtitleStyle()
        self._current_subtitle: Optional[TranscriptionResult] = None
        self._subtitle_expire_time: float = 0

    def render(
        self, 
        frame: np.ndarray, 
        result: Optional[TranscriptionResult],
        current_time: float,
        show_emotion: bool = True,
        show_intent: bool = False
    ) -> np.ndarray:
        """
        在帧上渲染字幕
        
        Args:
            frame: 视频帧 (BGR格式)
            result: 转录结果
            current_time: 当前时间
            show_emotion: 是否显示情感指示
            show_intent: 是否显示说话者意图
            
        Returns:
            渲染后的帧
        """
        # 更新当前字幕
        if result:
            self._current_subtitle = result
            self._subtitle_expire_time = result.segment.end_time + 2.0  # 字幕显示延长2秒
        
        # 检查字幕是否过期
        if current_time > self._subtitle_expire_time:
            self._current_subtitle = None
        
        if not self._current_subtitle:
            return frame
        
        # 复制帧以避免修改原始数据
        output = frame.copy()
        
        # 渲染字幕文本
        output = self._render_subtitle_text(output, self._current_subtitle)
        
        # 渲染情感指示器
        if show_emotion and self._current_subtitle.emotion:
            output = self._render_emotion_indicator(output, self._current_subtitle.emotion)
        
        # 渲染说话者意图
        if show_intent and self._current_subtitle.emotion:
            output = self._render_intent(output, self._current_subtitle.emotion)
        
        return output

    def _render_subtitle_text(
        self, 
        frame: np.ndarray, 
        result: TranscriptionResult
    ) -> np.ndarray:
        """渲染字幕文本"""
        text = result.segment.text
        if not text:
            return frame
        
        h, w = frame.shape[:2]
        
        # 获取情感对应的颜色
        if result.emotion:
            color = self.style.emotion_colors.get(
                result.emotion.emotion, 
                self.style.font_color
            )
        else:
            color = self.style.font_color
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.style.font_size / 30.0
        thickness = 2
        
        # 分行处理长文本
        max_chars_per_line = w // (self.style.font_size // 2) - 4
        lines = self._wrap_text(text, max_chars_per_line)
        
        # 计算总高度
        line_height = int(self.style.font_size * 1.5)
        total_height = len(lines) * line_height
        
        # 计算起始Y位置
        if self.style.position == "bottom":
            start_y = h - self.style.margin - total_height
        elif self.style.position == "top":
            start_y = self.style.margin + line_height
        else:  # center
            start_y = (h - total_height) // 2
        
        # 渲染每一行
        for i, line in enumerate(lines):
            y = start_y + i * line_height
            
            # 计算文本宽度以居中
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            x = (w - text_w) // 2
            
            # 绘制背景
            bg_padding = 10
            bg_rect = (
                x - bg_padding,
                y - text_h - bg_padding,
                x + text_w + bg_padding,
                y + bg_padding
            )
            self._draw_rounded_rect(
                frame, bg_rect, 
                self.style.bg_color, 
                self.style.bg_opacity,
                radius=10
            )
            
            # 绘制文本阴影
            cv2.putText(frame, line, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1)
            
            # 绘制文本
            cv2.putText(frame, line, (x, y), font, font_scale, color, thickness)
        
        return frame


    def _render_emotion_indicator(
        self, 
        frame: np.ndarray, 
        emotion: EmotionResult
    ) -> np.ndarray:
        """渲染情感指示器"""
        h, w = frame.shape[:2]
        
        # 在右上角显示情感信息
        emotion_name = EMOTION_NAMES.get(emotion.emotion, "未知")
        intensity_bar_width = int(100 * emotion.intensity)
        
        # 情感颜色
        color = self.style.emotion_colors.get(emotion.emotion, (255, 255, 255))
        
        # 绘制情感标签背景
        label_x = w - 180
        label_y = 20
        
        self._draw_rounded_rect(
            frame,
            (label_x, label_y, w - 20, label_y + 80),
            (30, 30, 30),
            0.8,
            radius=10
        )
        
        # 绘制情感名称
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame, 
            f"情感: {emotion_name}", 
            (label_x + 10, label_y + 25),
            font, 0.6, color, 1
        )
        
        # 绘制强度条
        bar_y = label_y + 40
        cv2.rectangle(frame, (label_x + 10, bar_y), (label_x + 10 + 100, bar_y + 10), (100, 100, 100), -1)
        cv2.rectangle(frame, (label_x + 10, bar_y), (label_x + 10 + intensity_bar_width, bar_y + 10), color, -1)
        
        # 绘制置信度
        cv2.putText(
            frame,
            f"置信度: {emotion.confidence:.0%}",
            (label_x + 10, label_y + 70),
            font, 0.5, (200, 200, 200), 1
        )
        
        return frame

    def _render_intent(
        self, 
        frame: np.ndarray, 
        emotion: EmotionResult
    ) -> np.ndarray:
        """渲染说话者意图"""
        h, w = frame.shape[:2]
        
        # 在左上角显示意图信息
        intent_text = f"意图: {emotion.speaker_intent}"
        
        # 绘制背景
        self._draw_rounded_rect(
            frame,
            (20, 20, 350, 50),
            (30, 30, 30),
            0.7,
            radius=8
        )
        
        # 绘制文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, intent_text, (30, 42), font, 0.5, (255, 255, 255), 1)
        
        return frame

    def _draw_rounded_rect(
        self,
        frame: np.ndarray,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        opacity: float,
        radius: int = 10
    ) -> None:
        """绘制圆角矩形"""
        x1, y1, x2, y2 = rect
        
        # 创建遮罩
        overlay = frame.copy()
        
        # 绘制圆角矩形
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # 绘制四个角的圆
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        
        # 混合
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """文本换行"""
        if len(text) <= max_chars:
            return [text]
        
        lines = []
        current_line = ""
        
        for char in text:
            current_line += char
            if len(current_line) >= max_chars:
                lines.append(current_line)
                current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def clear(self) -> None:
        """清除当前字幕"""
        self._current_subtitle = None
        self._subtitle_expire_time = 0
