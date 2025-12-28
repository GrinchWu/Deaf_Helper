"""
听障人群AI眼镜主动服务Agent
两阶段方案：本地快速检测 + 大模型精确分析

流程：
音频流 → 本地能量检测（毫秒级）→ [超阈值?] → 实时ASR识别 → 服务决策 → 输出
视频流 → 传入（不处理）→ 传递给后续模块
"""

import os
import cv2
import numpy as np
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

import dashscope
import pyaudio
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

# ==================== 配置 ====================
DASHSCOPE_API_KEY = "sk-471ef1d9f6734e79a4186adec3660bdd"
dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_websocket_api_url = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 800  # 100ms（原200ms，提速一倍）
ENERGY_THRESHOLD = 0.025  # 略微提高，减少误触发


# ==================== 数据模型 ====================
class ServiceType(Enum):
    NONE = "none"
    TRAFFIC_SAFETY = "traffic_safety"
    SOCIAL_TRANSCRIPTION = "social_transcription"
    EMERGENCY_ALERT = "emergency_alert"


SERVICE_NAMES = {
    ServiceType.NONE: "无需服务",
    ServiceType.TRAFFIC_SAFETY: "交通安全提醒",
    ServiceType.SOCIAL_TRANSCRIPTION: "社交语言转录",
    ServiceType.EMERGENCY_ALERT: "紧急警报"
}


# ==================== 服务决策器（大模型语义理解）====================
class ServiceDecider:
    """使用大模型理解语义，判断服务类型"""
    
    _client = None
    _cache = {}  # 简单缓存避免重复请求
    
    @classmethod
    def _get_client(cls):
        if cls._client is None:
            from openai import OpenAI
            cls._client = OpenAI(
                api_key=DASHSCOPE_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        return cls._client
    
    @classmethod
    def decide(cls, text: str) -> ServiceType:
        if not text or len(text) < 2:
            return ServiceType.NONE
        
        # 检查缓存
        if text in cls._cache:
            return cls._cache[text]
        
        try:
            client = cls._get_client()
            
            prompt = f"""你是听障人群的AI助手。分析以下语音内容，判断需要什么服务。

语音内容："{text}"

服务类型：
- emergency_alert: 紧急情况（求救、危险警告、事故、火灾、有人受伤、需要立即注意的威胁等）
- traffic_safety: 交通相关（车辆接近、交通信号、行人提醒、道路状况等）
- social_transcription: 社交对话（有人在和用户说话、打招呼、询问、日常交流等）
- none: 无关紧要的背景声音、噪音、无意义内容

只回复服务类型，不要其他内容："""

            response = client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # 解析结果
            if "emergency" in result:
                service = ServiceType.EMERGENCY_ALERT
            elif "traffic" in result:
                service = ServiceType.TRAFFIC_SAFETY
            elif "social" in result:
                service = ServiceType.SOCIAL_TRANSCRIPTION
            else:
                service = ServiceType.NONE
            
            # 缓存结果
            cls._cache[text] = service
            return service
            
        except Exception as e:
            # 大模型失败时回退到简单判断
            return cls._fallback_decide(text)
    
    @classmethod
    def _fallback_decide(cls, text: str) -> ServiceType:
        """回退方案：简单语义判断"""
        # 紧急相关
        if any(w in text for w in ['救', '帮', '危', '火', '伤', '痛', '急', '快']):
            return ServiceType.EMERGENCY_ALERT
        # 交通相关
        if any(w in text for w in ['车', '路', '走', '停', '让', '过']):
            return ServiceType.TRAFFIC_SAFETY
        # 有内容就是社交
        if len(text) > 2:
            return ServiceType.SOCIAL_TRANSCRIPTION
        return ServiceType.NONE


# ==================== ASR回调 ====================
class ASRCallback(RecognitionCallback):
    """实时语音识别回调"""
    
    def __init__(self, on_result: Callable[[str, bool], None]):
        self.on_result = on_result
    
    def on_open(self) -> None:
        pass
    
    def on_close(self) -> None:
        pass
    
    def on_complete(self) -> None:
        pass
    
    def on_error(self, message) -> None:
        print(f'ASR错误: {message.message}')
    
    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if 'text' in sentence:
            text = sentence['text']
            is_end = RecognitionResult.is_sentence_end(sentence)
            self.on_result(text, is_end)


# ==================== 主动服务Agent ====================
class ActiveServiceAgent:
    """两阶段：本地检测 + 大模型分析"""
    
    def __init__(self):
        self.running = False
        
        # 音频
        self.mic = None
        self.stream = None
        self.recognition = None
        self.asr_started = False
        
        # 视频
        self.cap = None
        self.last_frame: Optional[np.ndarray] = None
        
        # 状态
        self.last_service = ServiceType.NONE
        self.last_output_time = 0
        self.silence_count = 0
    
    def _calc_energy(self, audio_data: bytes) -> float:
        """计算音频能量"""
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        return np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0.0
    
    def _on_asr_result(self, text: str, is_end: bool):
        """ASR结果回调"""
        if text:
            service = ServiceDecider.decide(text)
            if service != ServiceType.NONE:
                self._output(service, text)
    
    def _output(self, service: ServiceType, text: str = ""):
        """输出服务决策"""
        now = time.time()
        # 防抖500ms
        if service == self.last_service and (now - self.last_output_time) < 0.5:
            return
        
        self.last_service = service
        self.last_output_time = now
        print(f"【需要服务】{SERVICE_NAMES[service]}")
    
    def _capture_video(self):
        """视频捕获"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
            time.sleep(0.033)
        self.cap.release()
    
    def _start_asr(self):
        """启动ASR"""
        if self.asr_started:
            return
        
        callback = ASRCallback(self._on_asr_result)
        self.recognition = Recognition(
            model='fun-asr-realtime-2025-11-07',
            format='wav',
            sample_rate=SAMPLE_RATE,
            semantic_punctuation_enabled=False,
            callback=callback
        )
        self.recognition.start()
        self.asr_started = True
    
    def _stop_asr(self):
        """停止ASR"""
        if not self.asr_started:
            return
        
        if self.recognition:
            self.recognition.stop()
            self.recognition = None
        self.asr_started = False
    
    def start(self):
        """启动Agent"""
        if not DASHSCOPE_API_KEY:
            print("错误: 请设置环境变量 DASHSCOPE_API_KEY")
            return
        
        self.running = True
        
        # 启动视频
        threading.Thread(target=self._capture_video, daemon=True).start()
        
        # 初始化音频
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("主动服务Agent已启动，按Ctrl+C停止...")
        
        try:
            while self.running:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                energy = self._calc_energy(data)
                
                if energy > ENERGY_THRESHOLD:
                    # 有声音：启动ASR并发送数据
                    self._start_asr()
                    if self.recognition:
                        self.recognition.send_audio_frame(data)
                    self.silence_count = 0
                else:
                    # 静音计数
                    self.silence_count += 1
                    if self.silence_count > 10:  # 连续2秒静音
                        self._stop_asr()
                        
        except KeyboardInterrupt:
            print("\n停止...")
        finally:
            self.stop()
    
    def stop(self):
        """停止"""
        self.running = False
        self._stop_asr()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.mic:
            self.mic.terminate()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        return self.last_frame


def main():
    agent = ActiveServiceAgent()
    agent.start()


if __name__ == "__main__":
    main()
