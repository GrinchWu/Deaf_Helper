"""
服务决策引擎 - 判断是否需要提供服务以及服务类型
支持社交语言转录和交通安全提醒两种服务
"""
from typing import Optional

from .models import (
    SceneAnalysisResult, ServiceDecision, ServiceType,
    UserLocation, UserState
)
from .memory_manager import MemoryManager


# 社交场景位置
SOCIAL_LOCATIONS = {
    UserLocation.MEETING_ROOM,
    UserLocation.BAR,
    UserLocation.RESTAURANT,
    UserLocation.OFFICE,
}

# 交通场景位置
TRAFFIC_LOCATIONS = {
    UserLocation.STREET,
    UserLocation.PUBLIC_TRANSPORT,
}

# 交通相关状态
TRAFFIC_STATES = {
    UserState.WALKING,
    UserState.CYCLING,
    UserState.DRIVING,
}


class ServiceDecisionEngine:
    """服务决策引擎"""

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager

    def _is_social_scenario(self, analysis: SceneAnalysisResult) -> bool:
        """
        判断是否为社交场景
        条件：用户在社交场所（会议室、酒吧、餐厅）或正在交谈
        """
        # 位置判断
        if analysis.user_location in SOCIAL_LOCATIONS:
            return True
        
        # 状态判断 - 正在交谈
        if analysis.user_state == UserState.TALKING:
            return True
        
        # 检查物体中是否有人（可能在交谈）
        social_objects = ["人", "person", "people", "同事", "朋友"]
        for obj in analysis.objects:
            obj_lower = obj.lower()
            for social_obj in social_objects:
                if social_obj in obj_lower:
                    # 如果有人且在可能的社交场所
                    if analysis.user_location not in TRAFFIC_LOCATIONS:
                        return True
        
        return False

    def _is_traffic_scenario(self, analysis: SceneAnalysisResult) -> bool:
        """
        判断是否为交通场景
        条件：用户在街道上且处于行走/骑行/驾驶状态
        """
        # 必须在街道或交通相关位置
        if analysis.user_location not in TRAFFIC_LOCATIONS:
            return False
        
        # 必须处于移动状态
        if analysis.user_state in TRAFFIC_STATES:
            return True
        
        # 检查物体中是否有交通相关物体
        traffic_objects = ["车", "car", "汽车", "红绿灯", "traffic", "斑马线", "行人"]
        for obj in analysis.objects:
            obj_lower = obj.lower()
            for traffic_obj in traffic_objects:
                if traffic_obj in obj_lower:
                    return True
        
        return False


    def _calculate_priority(self, analysis: SceneAnalysisResult, service_type: ServiceType) -> int:
        """
        计算服务优先级 (1-10)
        交通安全优先级更高
        """
        if service_type == ServiceType.TRAFFIC_SAFETY_ALERT:
            # 交通场景基础优先级较高
            base_priority = 7
            
            # 如果正在移动，优先级更高
            if analysis.user_state in {UserState.WALKING, UserState.CYCLING}:
                base_priority += 1
            if analysis.user_state == UserState.DRIVING:
                base_priority += 2
            
            return min(base_priority, 10)
        
        elif service_type == ServiceType.SOCIAL_TRANSCRIPTION:
            # 社交场景基础优先级
            base_priority = 5
            
            # 如果正在交谈，优先级更高
            if analysis.user_state == UserState.TALKING:
                base_priority += 2
            
            # 会议室场景优先级更高
            if analysis.user_location == UserLocation.MEETING_ROOM:
                base_priority += 1
            
            return min(base_priority, 10)
        
        return 1

    async def decide(self, analysis: SceneAnalysisResult) -> ServiceDecision:
        """
        根据场景分析结果做出服务决策
        
        Args:
            analysis: 场景分析结果
            
        Returns:
            ServiceDecision: 服务决策
        """
        # 判断场景类型
        is_traffic = self._is_traffic_scenario(analysis)
        is_social = self._is_social_scenario(analysis)
        
        # 确定服务类型（交通优先）
        if is_traffic:
            service_type = ServiceType.TRAFFIC_SAFETY_ALERT
            needs_service = True
            reason = f"用户在{analysis.user_location.value}，状态为{analysis.user_state.value}，检测到交通场景"
        elif is_social:
            service_type = ServiceType.SOCIAL_TRANSCRIPTION
            needs_service = True
            reason = f"用户在{analysis.user_location.value}，状态为{analysis.user_state.value}，检测到社交场景"
        else:
            service_type = ServiceType.NONE
            needs_service = False
            reason = "当前场景不需要特殊服务"
        
        # 计算优先级
        priority = self._calculate_priority(analysis, service_type) if needs_service else 0
        
        decision = ServiceDecision(
            frame_id=analysis.frame_id,
            timestamp=analysis.timestamp,
            needs_service=needs_service,
            service_type=service_type if needs_service else None,
            reason=reason,
            priority=priority,
        )
        
        return decision

    def decide_sync(self, analysis: SceneAnalysisResult) -> ServiceDecision:
        """
        同步版本的决策方法（用于测试）
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.decide(analysis))
