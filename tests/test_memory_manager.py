"""
MemoryManager 属性测试和单元测试
"""
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from src.models import (
    MemoryConfig, ContextEntry, SceneAnalysisResult, ServiceDecision,
    UserLocation, UserState, ServiceType
)
from src.memory_manager import MemoryManager


# Hypothesis strategies for generating test data
user_location_strategy = st.sampled_from(list(UserLocation))
user_state_strategy = st.sampled_from(list(UserState))
service_type_strategy = st.sampled_from(list(ServiceType))


@st.composite
def scene_analysis_strategy(draw):
    """生成随机SceneAnalysisResult"""
    frame_id = draw(st.integers(min_value=0, max_value=1000))
    return SceneAnalysisResult(
        frame_id=frame_id,
        timestamp=float(frame_id),
        scene_description=draw(st.text(alphabet=st.characters(whitelist_categories=('L', 'N')), min_size=1, max_size=50)),
        objects=draw(st.lists(st.text(alphabet=st.characters(whitelist_categories=('L',)), min_size=1, max_size=10), min_size=0, max_size=5)),
        user_location=draw(user_location_strategy),
        user_state=draw(user_state_strategy),
        confidence=draw(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)),
        raw_response="",
    )


@st.composite
def context_entry_strategy(draw):
    """生成随机ContextEntry"""
    analysis = draw(scene_analysis_strategy())
    return ContextEntry(
        frame_id=analysis.frame_id,
        timestamp=analysis.timestamp,
        analysis=analysis,
        decision=None,
    )


# Feature: hearing-impaired-agent, Property 12: Sliding Window Memory Limit
# Validates: Requirements 5.1, 5.2
class TestSlidingWindowMemoryLimit:
    """Property 12: 滑动窗口内存限制测试"""

    @given(
        max_frames=st.integers(min_value=5, max_value=50),
        num_entries=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    def test_sliding_window_never_exceeds_max(self, max_frames: int, num_entries: int):
        """
        For any sequence of N context entries stored in Memory_Manager where N > max_context_frames,
        the stored context SHALL contain exactly max_context_frames entries.
        """
        config = MemoryConfig(max_context_frames=max_frames)
        manager = MemoryManager(config)
        
        # 添加num_entries个条目
        for i in range(num_entries):
            entry = ContextEntry(
                frame_id=i,
                timestamp=float(i),
                analysis=SceneAnalysisResult(
                    frame_id=i,
                    timestamp=float(i),
                    scene_description=f"Scene {i}",
                    objects=["obj"],
                    user_location=UserLocation.UNKNOWN,
                    user_state=UserState.UNKNOWN,
                    confidence=0.5,
                ),
            )
            manager.store_context(entry)
        
        # 验证存储的条目数不超过max_frames
        assert manager.get_context_count() <= max_frames


    @given(
        max_frames=st.integers(min_value=5, max_value=20),
        num_entries=st.integers(min_value=30, max_value=100)
    )
    @settings(max_examples=100)
    def test_sliding_window_retains_most_recent(self, max_frames: int, num_entries: int):
        """
        For any sequence where N > max_context_frames, only the most recent entries are retained.
        """
        config = MemoryConfig(max_context_frames=max_frames)
        manager = MemoryManager(config)
        
        # 添加num_entries个条目
        for i in range(num_entries):
            entry = ContextEntry(
                frame_id=i,
                timestamp=float(i),
                analysis=SceneAnalysisResult(
                    frame_id=i,
                    timestamp=float(i),
                    scene_description=f"Scene {i}",
                    objects=[],
                    user_location=UserLocation.UNKNOWN,
                    user_state=UserState.UNKNOWN,
                    confidence=0.5,
                ),
            )
            manager.store_context(entry)
        
        # 获取所有存储的条目
        stored = manager.get_all_context()
        
        # 验证保留的是最近的条目
        if num_entries > max_frames:
            expected_start_id = num_entries - max_frames
            for i, entry in enumerate(stored):
                assert entry.frame_id == expected_start_id + i


# Feature: hearing-impaired-agent, Property 16: Context JSON Round-Trip
# Validates: Requirements 5.6
class TestContextJsonRoundTrip:
    """Property 16: 上下文JSON往返测试"""

    @given(entries=st.lists(context_entry_strategy(), min_size=0, max_size=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_context_json_roundtrip(self, entries):
        """
        For any valid ContextEntry list stored in Memory_Manager,
        serializing to JSON and deserializing back SHALL produce an equivalent list.
        """
        config = MemoryConfig(max_context_frames=100)
        manager = MemoryManager(config)
        
        # 存储所有条目
        for entry in entries:
            manager.store_context(entry)
        
        # 序列化
        serialized = manager.serialize()
        
        # 创建新的manager并反序列化
        new_manager = MemoryManager(config)
        new_manager.deserialize(serialized)
        
        # 验证条目数量相同
        assert manager.get_context_count() == new_manager.get_context_count()
        
        # 验证每个条目相同
        original = manager.get_all_context()
        restored = new_manager.get_all_context()
        
        for orig, rest in zip(original, restored):
            assert orig.frame_id == rest.frame_id
            assert orig.timestamp == rest.timestamp
            assert orig.analysis.scene_description == rest.analysis.scene_description
            assert orig.analysis.objects == rest.analysis.objects
            assert orig.analysis.user_location == rest.analysis.user_location
            assert orig.analysis.user_state == rest.analysis.user_state


# Unit Tests
class TestMemoryManagerUnit:
    """MemoryManager 单元测试"""

    def test_empty_buffer_operations(self):
        """测试空缓冲区操作"""
        manager = MemoryManager()
        
        assert manager.get_context_count() == 0
        assert manager.get_recent_context(10) == []
        assert manager.get_all_context() == []
        assert manager.get_latest_analysis() is None
        assert manager.get_latest_decision() is None

    def test_store_and_retrieve(self):
        """测试存储和检索"""
        manager = MemoryManager()
        
        entry = ContextEntry(
            frame_id=1,
            timestamp=1.0,
            analysis=SceneAnalysisResult(
                frame_id=1,
                timestamp=1.0,
                scene_description="Test scene",
                objects=["car", "person"],
                user_location=UserLocation.STREET,
                user_state=UserState.WALKING,
                confidence=0.9,
            ),
        )
        
        manager.store_context(entry)
        
        assert manager.get_context_count() == 1
        assert manager.get_latest_analysis().scene_description == "Test scene"

    def test_purge_old_context(self):
        """测试清理旧上下文"""
        config = MemoryConfig(max_context_frames=100)
        manager = MemoryManager(config)
        
        # 添加50个条目
        for i in range(50):
            entry = ContextEntry(
                frame_id=i,
                timestamp=float(i),
                analysis=SceneAnalysisResult(
                    frame_id=i,
                    timestamp=float(i),
                    scene_description=f"Scene {i}",
                    objects=[],
                    user_location=UserLocation.UNKNOWN,
                    user_state=UserState.UNKNOWN,
                    confidence=0.5,
                ),
            )
            manager.store_context(entry)
        
        # 清理，保留20个
        purged = manager.purge_old_context(keep_count=20)
        
        assert purged == 30
        assert manager.get_context_count() == 20
        # 验证保留的是最近的20个
        assert manager.get_all_context()[0].frame_id == 30
