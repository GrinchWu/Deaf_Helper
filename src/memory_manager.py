"""
记忆管理器 - 负责上下文存储和能耗优化
实现滑动窗口机制，支持JSON序列化/反序列化
"""
from collections import deque
from typing import List, Optional
import json
import sys

from .models import MemoryConfig, ContextEntry, SceneAnalysisResult, ServiceDecision


class MemoryManager:
    """记忆管理器"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._context_buffer: deque[ContextEntry] = deque(
            maxlen=self.config.max_context_frames
        )

    def store_context(self, entry: ContextEntry) -> None:
        """
        存储上下文条目
        使用deque的maxlen自动实现滑动窗口
        """
        self._context_buffer.append(entry)
        
        # 如果启用自动清理且超过内存阈值，执行清理
        if self.config.auto_purge:
            if self.get_memory_usage_mb() > self.config.memory_threshold_mb:
                self.purge_old_context()

    def get_recent_context(self, n: int = 10) -> List[ContextEntry]:
        """获取最近N个上下文条目"""
        if n <= 0:
            return []
        # 返回最近的n个条目
        context_list = list(self._context_buffer)
        return context_list[-n:] if len(context_list) >= n else context_list

    def get_all_context(self) -> List[ContextEntry]:
        """获取所有上下文条目"""
        return list(self._context_buffer)

    def get_context_count(self) -> int:
        """获取当前存储的上下文条目数量"""
        return len(self._context_buffer)

    def get_memory_usage_mb(self) -> float:
        """
        获取当前内存使用量(MB)
        使用sys.getsizeof估算内存占用
        """
        total_size = 0
        for entry in self._context_buffer:
            # 估算每个条目的大小
            total_size += sys.getsizeof(entry)
            total_size += sys.getsizeof(entry.analysis.scene_description)
            total_size += sys.getsizeof(entry.analysis.objects)
            total_size += sys.getsizeof(entry.analysis.raw_response)
        return total_size / (1024 * 1024)


    def purge_old_context(self, keep_count: Optional[int] = None) -> int:
        """
        清理旧的上下文数据
        
        Args:
            keep_count: 保留的条目数量，默认为max_context_frames的一半
            
        Returns:
            清理的条目数
        """
        if keep_count is None:
            keep_count = self.config.max_context_frames // 2
        
        original_count = len(self._context_buffer)
        
        if original_count <= keep_count:
            return 0
        
        # 保留最近的keep_count个条目
        recent_entries = list(self._context_buffer)[-keep_count:]
        self._context_buffer.clear()
        for entry in recent_entries:
            self._context_buffer.append(entry)
        
        return original_count - len(self._context_buffer)

    def clear(self) -> None:
        """清空所有上下文"""
        self._context_buffer.clear()

    def serialize(self) -> str:
        """序列化上下文为JSON"""
        entries = [entry.to_dict() for entry in self._context_buffer]
        return json.dumps(entries, ensure_ascii=False)

    def deserialize(self, data: str) -> None:
        """从JSON反序列化上下文"""
        self._context_buffer.clear()
        entries = json.loads(data)
        for entry_dict in entries:
            entry = ContextEntry.from_dict(entry_dict)
            self._context_buffer.append(entry)

    def save_to_file(self, filepath: str) -> None:
        """保存上下文到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.serialize())

    def load_from_file(self, filepath: str) -> None:
        """从文件加载上下文"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.deserialize(f.read())

    def get_latest_analysis(self) -> Optional[SceneAnalysisResult]:
        """获取最新的场景分析结果"""
        if not self._context_buffer:
            return None
        return self._context_buffer[-1].analysis

    def get_latest_decision(self) -> Optional[ServiceDecision]:
        """获取最新的服务决策"""
        if not self._context_buffer:
            return None
        return self._context_buffer[-1].decision
