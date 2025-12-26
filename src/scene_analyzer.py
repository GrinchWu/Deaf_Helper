"""
场景分析器 - 调用VLLM模型分析场景
输出结构化的场景信息：描述、物体、位置、状态
"""
import json
import logging
import re
from typing import Optional

from .models import (
    FrameData, SceneAnalysisResult, UserLocation, UserState
)
from .vllm_client import VLLMClient, VLLMRequest

logger = logging.getLogger(__name__)


# 位置关键词映射
LOCATION_KEYWORDS = {
    UserLocation.MEETING_ROOM: ["会议室", "会议", "meeting room", "conference"],
    UserLocation.BAR: ["酒吧", "bar", "pub", "夜店"],
    UserLocation.STREET: ["街道", "马路", "道路", "十字路口", "人行道", "street", "road", "crosswalk"],
    UserLocation.RESTAURANT: ["餐厅", "饭店", "餐馆", "食堂", "restaurant", "cafe", "咖啡"],
    UserLocation.HOME: ["家", "客厅", "卧室", "home", "living room", "bedroom"],
    UserLocation.OFFICE: ["办公室", "工位", "office", "desk", "workspace"],
    UserLocation.SHOPPING_MALL: ["商场", "购物中心", "超市", "mall", "shopping", "store"],
    UserLocation.PUBLIC_TRANSPORT: ["地铁", "公交", "火车", "bus", "subway", "train", "metro"],
}

# 状态关键词映射
STATE_KEYWORDS = {
    UserState.WALKING: ["走路", "行走", "步行", "walking", "walk"],
    UserState.CYCLING: ["骑车", "骑行", "自行车", "cycling", "bike", "bicycle"],
    UserState.DRIVING: ["开车", "驾驶", "driving", "drive", "car"],
    UserState.SITTING: ["坐", "sitting", "sit", "seated"],
    UserState.STANDING: ["站", "standing", "stand"],
    UserState.TALKING: ["交谈", "说话", "聊天", "talking", "speaking", "conversation"],
}


class SceneAnalyzer:
    """场景分析器"""

    def __init__(self, vllm_client: VLLMClient):
        self.vllm_client = vllm_client
        self._last_prompt: Optional[str] = None

    def build_analysis_prompt(self) -> str:
        """
        构建分析提示词
        包含场景描述、物体、位置、状态的请求
        """
        prompt = """请分析这张图片，并以JSON格式输出以下信息：

1. scene_description: 简单描述这个画面（一句话）
2. objects: 画面中出现了什么物体（列表形式）
3. user_location: 用户处于什么位置，从以下选项中选择：
   - meeting_room (会议室)
   - bar (酒吧)
   - street (街道/马路)
   - restaurant (餐厅/饭店)
   - home (家)
   - office (办公室)
   - shopping_mall (商场)
   - public_transport (公共交通)
   - unknown (未知)

4. user_state: 用户处于什么状态，从以下选项中选择：
   - walking (走路)
   - cycling (骑行)
   - driving (开车)
   - sitting (坐着)
   - standing (站着)
   - talking (交谈)
   - unknown (未知)

请直接输出JSON，不要添加其他说明文字。示例格式：
{
  "scene_description": "一个繁忙的十字路口",
  "objects": ["汽车", "行人", "红绿灯"],
  "user_location": "street",
  "user_state": "walking"
}"""
        self._last_prompt = prompt
        return prompt


    def _parse_location(self, location_str: str) -> UserLocation:
        """解析位置字符串为枚举值"""
        location_str = location_str.lower().strip()
        
        # 直接匹配枚举值
        for loc in UserLocation:
            if loc.value == location_str:
                return loc
        
        # 关键词匹配
        for loc, keywords in LOCATION_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in location_str:
                    return loc
        
        return UserLocation.UNKNOWN

    def _parse_state(self, state_str: str) -> UserState:
        """解析状态字符串为枚举值"""
        state_str = state_str.lower().strip()
        
        # 直接匹配枚举值
        for state in UserState:
            if state.value == state_str:
                return state
        
        # 关键词匹配
        for state, keywords in STATE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in state_str:
                    return state
        
        return UserState.UNKNOWN

    def _extract_json_from_response(self, response: str) -> dict:
        """从响应中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # 尝试提取更复杂的JSON（包含嵌套）
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return {}

    async def analyze_frame(self, frame: FrameData) -> SceneAnalysisResult:
        """
        分析单帧图像
        
        Args:
            frame: 帧数据
            
        Returns:
            SceneAnalysisResult: 场景分析结果
        """
        prompt = self.build_analysis_prompt()
        request = VLLMRequest(image_data=frame.image_data, prompt=prompt)
        
        try:
            # 收集流式响应
            response_parts = []
            async for chunk in self.vllm_client.analyze_stream(request):
                if not chunk.is_final:
                    response_parts.append(chunk.content)
            
            raw_response = "".join(response_parts)
            
            # 解析响应
            parsed = self._extract_json_from_response(raw_response)
            
            return SceneAnalysisResult(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                scene_description=parsed.get("scene_description", "无法识别场景"),
                objects=parsed.get("objects", []),
                user_location=self._parse_location(parsed.get("user_location", "unknown")),
                user_state=self._parse_state(parsed.get("user_state", "unknown")),
                confidence=0.8 if parsed else 0.3,
                raw_response=raw_response,
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze frame {frame.frame_id}: {e}")
            # 返回默认结果
            return SceneAnalysisResult(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                scene_description="分析失败",
                objects=[],
                user_location=UserLocation.UNKNOWN,
                user_state=UserState.UNKNOWN,
                confidence=0.0,
                raw_response=str(e),
            )

    def get_last_prompt(self) -> Optional[str]:
        """获取最后使用的提示词（用于测试）"""
        return self._last_prompt
