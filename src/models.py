"""
核心数据模型和枚举类型定义
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict
from enum import Enum
import json


class UserLocation(Enum):
    """用户位置类型"""
    MEETING_ROOM = "meeting_room"
    BAR = "bar"
    STREET = "street"
    RESTAURANT = "restaurant"
    HOME = "home"
    OFFICE = "office"
    SHOPPING_MALL = "shopping_mall"
    PUBLIC_TRANSPORT = "public_transport"
    UNKNOWN = "unknown"


class UserState(Enum):
    """用户状态类型"""
    WALKING = "walking"
    CYCLING = "cycling"
    DRIVING = "driving"
    SITTING = "sitting"
    STANDING = "standing"
    TALKING = "talking"
    UNKNOWN = "unknown"


class ServiceType(Enum):
    """服务类型"""
    SOCIAL_TRANSCRIPTION = "social_transcription"  # 社交语言转录文字
    TRAFFIC_SAFETY_ALERT = "traffic_safety_alert"  # 交通场景安全提醒
    NONE = "none"


@dataclass
class FrameData:
    """视频帧数据"""
    frame_id: int
    timestamp: float
    image_data: str  # Base64编码的图像数据
    width: int
    height: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameData":
        return cls(**data)


@dataclass
class ProcessorConfig:
    """视频处理器配置"""
    sample_rate: int = 1  # 每N帧采样一次
    max_fps: float = 30.0  # 最大处理帧率
    skip_similar_frames: bool = True  # 跳过相似帧
    similarity_threshold: float = 0.95  # 相似度阈值

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessorConfig":
        return cls(**data)


@dataclass
class VLLMConfig:
    """VLLM客户端配置"""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model: str = "qwen-omni-turbo"
    max_retries: int = 3
    timeout: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLLMConfig":
        return cls(**data)


@dataclass
class MemoryConfig:
    """记忆管理器配置"""
    max_context_frames: int = 100  # 最大存储帧数
    memory_threshold_mb: float = 100.0  # 内存阈值(MB)
    auto_purge: bool = True  # 自动清理
    persistence_path: Optional[str] = None  # 持久化路径

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        return cls(**data)


@dataclass
class SceneAnalysisResult:
    """场景分析结果"""
    frame_id: int
    timestamp: float
    scene_description: str  # 简单描述画面
    objects: List[str]  # 画面中出现的物体
    user_location: UserLocation  # 用户位置
    user_state: UserState  # 用户状态
    confidence: float  # 分析置信度
    raw_response: str = ""  # 原始VLLM响应

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "scene_description": self.scene_description,
            "objects": self.objects,
            "user_location": self.user_location.value,
            "user_state": self.user_state.value,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneAnalysisResult":
        return cls(
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            scene_description=data["scene_description"],
            objects=data["objects"],
            user_location=UserLocation(data["user_location"]),
            user_state=UserState(data["user_state"]),
            confidence=data["confidence"],
            raw_response=data.get("raw_response", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "SceneAnalysisResult":
        return cls.from_dict(json.loads(json_str))


@dataclass
class ServiceDecision:
    """服务决策结果"""
    frame_id: int
    timestamp: float
    needs_service: bool  # 是否需要服务
    service_type: Optional[ServiceType]  # 服务类型
    reason: str  # 决策原因
    priority: int  # 优先级 (1-10)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "needs_service": self.needs_service,
            "service_type": self.service_type.value if self.service_type else None,
            "reason": self.reason,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceDecision":
        service_type = None
        if data.get("service_type"):
            service_type = ServiceType(data["service_type"])
        return cls(
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            needs_service=data["needs_service"],
            service_type=service_type,
            reason=data["reason"],
            priority=data["priority"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ServiceDecision":
        return cls.from_dict(json.loads(json_str))

    def needs_service_str(self) -> str:
        """返回是否需要服务的中文表示"""
        return "是" if self.needs_service else "否"


@dataclass
class ContextEntry:
    """上下文条目"""
    frame_id: int
    timestamp: float
    analysis: SceneAnalysisResult
    decision: Optional[ServiceDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "analysis": self.analysis.to_dict(),
            "decision": self.decision.to_dict() if self.decision else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextEntry":
        decision = None
        if data.get("decision"):
            decision = ServiceDecision.from_dict(data["decision"])
        return cls(
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            analysis=SceneAnalysisResult.from_dict(data["analysis"]),
            decision=decision,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ContextEntry":
        return cls.from_dict(json.loads(json_str))


@dataclass
class AgentConfig:
    """Agent配置"""
    processor_config: ProcessorConfig = field(default_factory=ProcessorConfig)
    vllm_config: VLLMConfig = field(default_factory=VLLMConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    low_activity_threshold: float = 0.3  # 低活动阈值
    adaptive_sampling: bool = True  # 自适应采样

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processor_config": self.processor_config.to_dict(),
            "vllm_config": self.vllm_config.to_dict(),
            "memory_config": self.memory_config.to_dict(),
            "low_activity_threshold": self.low_activity_threshold,
            "adaptive_sampling": self.adaptive_sampling,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        return cls(
            processor_config=ProcessorConfig.from_dict(data.get("processor_config", {})),
            vllm_config=VLLMConfig.from_dict(data.get("vllm_config", {})),
            memory_config=MemoryConfig.from_dict(data.get("memory_config", {})),
            low_activity_threshold=data.get("low_activity_threshold", 0.3),
            adaptive_sampling=data.get("adaptive_sampling", True),
        )


@dataclass
class AgentOutput:
    """Agent输出"""
    frame_id: int
    timestamp: float
    scene_analysis: SceneAnalysisResult
    service_decision: ServiceDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "scene_analysis": self.scene_analysis.to_dict(),
            "service_decision": self.service_decision.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
