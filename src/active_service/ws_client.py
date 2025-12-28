"""
本地客户端 - 整合主动服务检测 + WebSocket远程调用 + 转录 + ToM分析

流程:
1. 启动主动服务检测，收集图像和音频流
2. 当检测到"【需要服务】社交语言转录"时，触发远程处理
3. 将图像流发送到远程服务器进行YOLOv5检测和EZ-VSL音源定位
4. 接收检测结果（包含speaker_id），实时显示转录和ToM分析结果

使用方法:
1. 远程服务器启动: conda activate ezvsl && python ws_server.py --port 8765
2. 本地SSH端口转发: ssh -p 31801 -L 8765:localhost:8765 root@connect.nmb1.seetacloud.com
3. 运行: python ws_client.py
"""

import asyncio
import json
import base64
import cv2
import numpy as np
import threading
import time
import os
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

try:
    import websockets
except ImportError:
    print("请安装: pip install websockets")
    exit(1)

# 添加transcription模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'transcription'))

# 直接导入main.py的组件
from main import (
    ActiveServiceAgent, 
    ServiceType, 
    SERVICE_NAMES,
    SAMPLE_RATE,
    CHUNK_SIZE,
    ENERGY_THRESHOLD
)

# 导入ToM分析器
try:
    from models import TranscriptSegment, EmotionResult
    from tom_analyzer import SimToMAnalyzer, ToMConfig
    TOM_AVAILABLE = True
except ImportError:
    TOM_AVAILABLE = False
    print("警告: ToM分析模块未找到，将跳过情感分析")

# 用于运行异步ToM分析
import nest_asyncio
try:
    nest_asyncio.apply()
except:
    pass

# 导入前端服务器
try:
    from ws_frontend_server import get_frontend_server, FrontendServer
    FRONTEND_AVAILABLE = True
except ImportError:
    FRONTEND_AVAILABLE = False
    print("警告: 前端服务模块未找到")

# ==================== 配置 ====================
REMOTE_SERVER_URL = "ws://localhost:8765"
STEP_DURATION = 0.5  # 每个step的时长（秒）
OUTPUT_DIR = "output_steps"  # 本地输出目录
DASHSCOPE_API_KEY = "sk-471ef1d9f6734e79a4186adec3660bdd"


# ==================== 数据模型 ====================
@dataclass
class PersonDetection:
    """人物检测结果"""
    label: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float


@dataclass
class StepResult:
    """单个step的处理结果"""
    step: int
    timestamp: float
    persons: List[PersonDetection] = field(default_factory=list)
    speaker_id: Optional[str] = None
    count: int = 0


@dataclass
class FrameData:
    """帧数据"""
    frame: np.ndarray
    timestamp: float
    step: int


# ==================== 远程服务客户端 ====================
class RemoteClient:
    """WebSocket客户端 - 支持流式处理"""
    
    def __init__(self, server_url: str = REMOTE_SERVER_URL):
        self.server_url = server_url
        self.ws = None
        self.loop = None
        self.connected = False
        self.current_step = 0
        self.results: Dict[int, StepResult] = {}
        self.result_callbacks = []
        
        # 音频缓冲区
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * 10))  # 10秒缓冲
        
    def add_result_callback(self, callback):
        """添加结果回调"""
        self.result_callbacks.append(callback)
    
    def connect(self):
        """连接远程服务器"""
        try:
            self.loop = asyncio.new_event_loop()
            self.ws = self.loop.run_until_complete(
                websockets.connect(self.server_url, max_size=10*1024*1024)
            )
            self.connected = True
            print(f"[WebSocket] 已连接远程服务器: {self.server_url}")
            return True
        except Exception as e:
            print(f"[WebSocket] 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.ws and self.loop:
            try:
                self.loop.run_until_complete(self.ws.close())
            except:
                pass
        self.connected = False
        print("[WebSocket] 已断开连接")
    
    def start_streaming_session(self) -> bool:
        """开始流式会话"""
        if not self.connected:
            return False
        
        try:
            request = {
                "command": "start_session",
                "data": {
                    "step_duration": STEP_DURATION,
                    "timestamp": datetime.now().isoformat()
                }
            }
            self.loop.run_until_complete(self.ws.send(json.dumps(request)))
            
            response = self.loop.run_until_complete(
                asyncio.wait_for(self.ws.recv(), timeout=10.0)
            )
            result = json.loads(response)
            
            if result.get("status") == "success":
                self.current_step = 0
                print("[WebSocket] 流式会话已开始")
                return True
            else:
                print(f"[WebSocket] 开始会话失败: {result.get('error')}")
                return False
                
        except Exception as e:
            print(f"[WebSocket] 开始会话异常: {e}")
            return False
    
    def send_frame(self, image: np.ndarray, audio_data: Optional[bytes] = None) -> Optional[StepResult]:
        """
        发送单帧图像和音频数据进行检测
        
        Args:
            image: 图像帧
            audio_data: 音频数据（可选）
        
        Returns:
            StepResult 或 None
        """
        if not self.connected:
            return None
        
        try:
            self.current_step += 1
            timestamp = self.current_step * STEP_DURATION
            
            # 编码图像
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 编码音频（如果有）
            audio_b64 = None
            if audio_data:
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 构建请求
            request = {
                "command": "process_frame",
                "data": {
                    "step": self.current_step,
                    "timestamp": timestamp,
                    "image": image_b64,
                    "audio": audio_b64
                }
            }
            
            self.loop.run_until_complete(self.ws.send(json.dumps(request)))
            
            # 接收响应
            response = self.loop.run_until_complete(
                asyncio.wait_for(self.ws.recv(), timeout=30.0)
            )
            result = json.loads(response)
            
            if result.get("status") == "success":
                # 解析结果
                step_result = StepResult(
                    step=self.current_step,
                    timestamp=timestamp,
                    persons=[
                        PersonDetection(
                            label=p["label"],
                            bbox=p["bbox"],
                            confidence=p["confidence"]
                        )
                        for p in result.get("persons", [])
                    ],
                    speaker_id=result.get("speaker_id"),
                    count=result.get("count", 0)
                )
                
                self.results[self.current_step] = step_result
                
                # 触发回调
                for callback in self.result_callbacks:
                    callback(step_result)
                
                return step_result
            else:
                print(f"[WebSocket] 处理失败: {result.get('error')}")
                return None
                
        except Exception as e:
            print(f"[WebSocket] 发送帧异常: {e}")
            return None
    
    def end_streaming_session(self) -> Dict[str, Any]:
        """结束流式会话，获取汇总结果"""
        if not self.connected:
            return {}
        
        try:
            request = {
                "command": "end_session",
                "data": {}
            }
            self.loop.run_until_complete(self.ws.send(json.dumps(request)))
            
            response = self.loop.run_until_complete(
                asyncio.wait_for(self.ws.recv(), timeout=30.0)
            )
            result = json.loads(response)
            
            print(f"[WebSocket] 会话结束，共处理 {self.current_step} 个step")
            return result
            
        except Exception as e:
            print(f"[WebSocket] 结束会话异常: {e}")
            return {}
    
    def detect_person(self, image: np.ndarray) -> Optional[dict]:
        """单次人物检测（兼容旧接口）"""
        if not self.connected:
            return None
        try:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            request = {"command": "detect_person", "data": {"image": image_b64}}
            self.loop.run_until_complete(self.ws.send(json.dumps(request)))
            
            response = self.loop.run_until_complete(
                asyncio.wait_for(self.ws.recv(), timeout=10.0)
            )
            return json.loads(response)
        except Exception as e:
            print(f"[WebSocket] 远程调用失败: {e}")
            return None


# ==================== 结果管理器 ====================
class ResultManager:
    """管理和保存处理结果"""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.all_results: List[StepResult] = []
    
    def save_step_result(self, result: StepResult):
        """保存单个step的结果（静默保存，不打印）"""
        self.all_results.append(result)
        
        # 构建输出JSON
        output_data = {
            "step": result.step,
            "timestamp": result.timestamp,
            "persons": [
                {
                    "label": p.label,
                    "bbox": p.bbox,
                    "confidence": p.confidence
                }
                for p in result.persons
            ],
            "speaker_id": result.speaker_id,
            "count": result.count
        }
        
        # 保存到文件
        filename = f"output_step_{result.step}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def save_summary(self):
        """保存汇总结果"""
        # 构建frames数组
        frames = []
        for result in self.all_results:
            frame_data = {
                "frame": int(result.timestamp * 30),  # 假设30fps
                "timestamp": result.timestamp,
                "persons": [
                    {
                        "label": p.label,
                        "bbox": p.bbox,
                        "confidence": p.confidence
                    }
                    for p in result.persons
                ],
                "speaker_id": result.speaker_id,
                "count": result.count
            }
            frames.append(frame_data)
        
        summary = {
            "total_steps": len(self.all_results),
            "frames": frames
        }
        
        filepath = os.path.join(self.output_dir, "summary.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def get_results_for_transcription(self) -> List[Dict]:
        """获取用于转录的结果列表"""
        return [
            {
                "step": r.step,
                "timestamp": r.timestamp,
                "speaker_id": r.speaker_id,
                "persons": [
                    {"label": p.label, "bbox": p.bbox}
                    for p in r.persons
                ]
            }
            for r in self.all_results
        ]


# ==================== 扩展Agent（继承main.py的Agent）====================
class ExtendedAgent(ActiveServiceAgent):
    """扩展Agent，添加远程调用和流式处理功能"""
    
    def __init__(self):
        super().__init__()
        self.remote_client = RemoteClient()
        self.result_manager = ResultManager()
        self.pending_text = ""
        
        # 流式处理状态
        self.streaming_active = False
        self.last_frame_time = 0
        self.frame_buffer: deque = deque(maxlen=100)
        self.audio_buffer: deque = deque(maxlen=int(SAMPLE_RATE * 10))
        
        # 音频收集锁
        self._audio_lock = threading.Lock()
        
        # 当前说话人（用于实时转录显示）
        self.current_speaker_id: Optional[str] = None
        
        # 当前检测到的人物列表
        self.current_persons: List[Dict] = []
        
        # 添加结果回调
        self.remote_client.add_result_callback(self._on_step_result)
        
        # ToM分析器
        self.tom_analyzer: Optional[SimToMAnalyzer] = None
        self.conversation_context: List[str] = []  # 对话上下文
        if TOM_AVAILABLE:
            config = ToMConfig(api_key=DASHSCOPE_API_KEY)
            self.tom_analyzer = SimToMAnalyzer(config)
        
        # 前端服务器
        self.frontend_server: Optional[FrontendServer] = None
        if FRONTEND_AVAILABLE:
            self.frontend_server = get_frontend_server()
    
    def _on_step_result(self, result: StepResult):
        """处理远程返回的step结果（静默保存，不打印）"""
        self.result_manager.save_step_result(result)
        # 更新当前说话人
        if result.speaker_id:
            self.current_speaker_id = result.speaker_id
        
        # 更新当前人物列表
        self.current_persons = [
            {
                "label": p.label,
                "bbox": p.bbox,
                "confidence": p.confidence
            }
            for p in result.persons
        ]
        
        # 推送到前端
        if self.frontend_server:
            self.frontend_server.send_detection(
                self.current_persons,
                result.speaker_id
            )
    
    def _on_asr_result(self, text: str, is_end: bool):
        """重写ASR回调 - 实时显示转录结果，ToM异步分析"""
        if text:
            # 获取当前说话人
            speaker = self.current_speaker_id or "未知"
            
            if is_end:
                # 句子结束，立即显示转录（不等待ToM）
                print(f"[{speaker}] {text}")
                
                # 推送到前端
                if self.frontend_server:
                    self.frontend_server.send_transcript(speaker, text)
                
                # 添加到对话上下文
                self.conversation_context.append(f"[{speaker}]: {text}")
                # 保留最近10轮对话
                if len(self.conversation_context) > 10:
                    self.conversation_context = self.conversation_context[-10:]
                
                # 每3句话触发一次ToM分析（异步，不阻塞）
                if len(self.conversation_context) % 3 == 0 and self.tom_analyzer and TOM_AVAILABLE:
                    threading.Thread(
                        target=self._run_tom_analysis_async,
                        args=(list(self.conversation_context),),
                        daemon=True
                    ).start()
            
            self.pending_text = text
            from main import ServiceDecider
            service = ServiceDecider.decide(text)
            if service != ServiceType.NONE:
                self._trigger_service(service, text)
    
    def _run_tom_analysis_async(self, context_snapshot: List[str]):
        """异步运行ToM分析（在独立线程中）"""
        try:
            # 取最后一句进行分析
            if not context_snapshot:
                return
            
            last_utterance = context_snapshot[-1]
            # 解析 "[speaker]: text" 格式
            if "]: " in last_utterance:
                speaker = last_utterance.split("]: ")[0].strip("[")
                text = last_utterance.split("]: ", 1)[1]
            else:
                speaker = "未知"
                text = last_utterance
            
            # 创建TranscriptSegment
            segment = TranscriptSegment(
                text=text,
                start_time=0,
                end_time=0,
                speaker=speaker
            )
            
            # 构建上下文（排除最后一句）
            context = "\n".join(context_snapshot[:-1]) if len(context_snapshot) > 1 else ""
            
            # 每次创建新的分析器实例，避免事件循环关闭问题
            config = ToMConfig(api_key=DASHSCOPE_API_KEY)
            analyzer = SimToMAnalyzer(config)
            
            # 在独立事件循环中运行异步分析
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    analyzer.analyze(segment, context)
                )
                
                # 显示ToM分析结果
                if result:
                    print(f"\n  ═══ ToM分析 ═══")
                    print(f"  → 情感: {result.emotion.value} ({result.confidence:.0%})")
                    if result.speaker_intent and result.speaker_intent != "无法推断":
                        print(f"  → 意图: {result.speaker_intent}")
                    if result.mental_state and result.mental_state != "无法推断":
                        print(f"  → 心理: {result.mental_state}")
                    if result.suggested_response:
                        print(f"  → 建议: {result.suggested_response}")
                    print()
                    
                    # 推送到前端
                    if self.frontend_server:
                        print(f"  [DEBUG] 正在推送ToM到前端...")
                        self.frontend_server.send_tom(
                            emotion=result.emotion.value,
                            confidence=result.confidence,
                            intent=result.speaker_intent,
                            mental_state=result.mental_state,
                            suggested_response=result.suggested_response
                        )
                        print(f"  [DEBUG] ToM推送完成")
                    else:
                        print(f"  [DEBUG] frontend_server 为 None，无法推送ToM")
                        
                # 关闭分析器的HTTP客户端
                loop.run_until_complete(analyzer.close())
            finally:
                loop.close()
                
        except Exception as e:
            # ToM分析失败不影响主流程
            print(f"  [DEBUG] ToM分析异常: {e}")
    
    def _trigger_service(self, service: ServiceType, text: str):
        """触发服务（带远程调用）"""
        now = time.time()
        if service == self.last_service and (now - self.last_output_time) < 0.5:
            return
        
        self.last_service = service
        self.last_output_time = now
        
        # 社交语言转录时，启动流式处理（静默启动）
        if service == ServiceType.SOCIAL_TRANSCRIPTION:
            self._start_streaming_process(text)
    
    def _start_streaming_process(self, text: str):
        """启动流式处理流程"""
        if self.streaming_active:
            return
        
        if not self.remote_client.connected:
            return
        
        self.streaming_active = True
        
        # 开始远程会话
        if not self.remote_client.start_streaming_session():
            self.streaming_active = False
            return
        
        # 启动流式发送线程
        threading.Thread(target=self._streaming_loop, daemon=True).start()
    
    def _streaming_loop(self):
        """流式处理循环"""
        try:
            while self.streaming_active and self.running:
                current_time = time.time()
                
                # 每0.5秒发送一帧
                if current_time - self.last_frame_time >= STEP_DURATION:
                    self.last_frame_time = current_time
                    
                    if self.last_frame is not None:
                        frame_copy = self.last_frame.copy()
                        
                        # 收集音频数据
                        audio_data = self._get_audio_chunk()
                        
                        # 发送帧到远程服务器（静默处理）
                        self.remote_client.send_frame(frame_copy, audio_data)
                        
                        # 推送视频帧到前端
                        if self.frontend_server:
                            _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            image_b64 = base64.b64encode(buffer).decode('utf-8')
                            self.frontend_server.send_frame(image_b64)
                
                time.sleep(0.05)  # 50ms检查间隔
                
        except Exception as e:
            print(f"[错误] 流式处理异常: {e}")
        finally:
            self._stop_streaming_process()
    
    def _get_audio_chunk(self) -> Optional[bytes]:
        """获取音频数据块"""
        with self._audio_lock:
            if len(self.audio_buffer) == 0:
                return None
            
            # 获取0.5秒的音频数据 (16000 * 0.5 = 8000 samples = 16000 bytes)
            samples_needed = int(SAMPLE_RATE * STEP_DURATION)
            audio_samples = []
            
            # 从buffer中取出数据
            bytes_needed = samples_needed * 2  # 16-bit = 2 bytes per sample
            collected = 0
            
            while collected < bytes_needed and len(self.audio_buffer) > 0:
                chunk = self.audio_buffer.popleft()
                audio_samples.append(chunk)
                collected += len(chunk)
            
            if audio_samples:
                return b''.join(audio_samples)
            return None
    
    def _collect_audio(self, audio_data: bytes):
        """收集音频数据到buffer"""
        with self._audio_lock:
            self.audio_buffer.append(audio_data)
    
    def _stop_streaming_process(self):
        """停止流式处理"""
        if not self.streaming_active:
            return
        
        self.streaming_active = False
        
        # 结束远程会话
        self.remote_client.end_streaming_session()
        
        # 保存汇总
        self.result_manager.save_summary()
    
    def _trigger_transcription(self):
        """触发本地转录服务（已废弃，转录现在是实时的）"""
        pass
    
    def _call_remote(self, text: str):
        """调用远程人物检测（单次，兼容旧接口）"""
        if self.last_frame is None:
            print("  [远程] 无图像")
            return
        
        if not self.remote_client.connected:
            print("  [远程] 未连接")
            return
        
        print("  [远程] 检测人物...")
        result = self.remote_client.detect_person(self.last_frame)
        
        if result and result.get("status") == "success":
            persons = result.get("persons", [])
            print(f"  [远程] 检测到 {len(persons)} 人:")
            for p in persons:
                print(f"    - {p['label']}: bbox={p['bbox']}, conf={p['confidence']}")
            
            # 输出JSON
            output = {
                "service": "social_transcription",
                "text": text,
                "persons": persons
            }
            print(f"  [输出] {json.dumps(output, ensure_ascii=False)}")
        else:
            print(f"  [远程] 失败: {result.get('error') if result else '无响应'}")
    
    def start(self):
        """启动（先连接远程，然后运行主循环并收集音频）"""
        print("听障辅助转录系统启动中...")
        
        # 连接远程服务器
        remote_connected = self.remote_client.connect()
        if not remote_connected:
            print("警告: 远程服务器未连接，说话人识别功能不可用")
        
        # 推送状态到前端
        if self.frontend_server:
            self.frontend_server.send_status(remote=remote_connected, asr=False)
        
        # 以下是从父类复制并修改的start逻辑
        from main import DASHSCOPE_API_KEY, CHANNELS, CHUNK_SIZE, ENERGY_THRESHOLD
        import pyaudio
        
        if not DASHSCOPE_API_KEY:
            print("错误: 请设置环境变量 DASHSCOPE_API_KEY")
            return
        
        self.running = True
        
        # 启动视频
        threading.Thread(target=self._capture_video, daemon=True).start()
        
        # 启动前端视频推送线程（即使没有触发社交转录也推送视频）
        threading.Thread(target=self._push_video_to_frontend, daemon=True).start()
        
        # 初始化音频
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("系统就绪，开始监听...")
        print("-" * 40)
        
        try:
            while self.running:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                energy = self._calc_energy(data)
                
                # 收集音频数据到buffer（用于远程发送）
                self._collect_audio(data)
                
                if energy > ENERGY_THRESHOLD:
                    # 有声音：启动ASR并发送数据
                    self._start_asr()
                    if self.recognition:
                        self.recognition.send_audio_frame(data)
                    self.silence_count = 0
                    
                    # 更新ASR状态
                    if self.frontend_server:
                        self.frontend_server.send_status(asr=True)
                else:
                    # 静音计数
                    self.silence_count += 1
                    if self.silence_count > 10:  # 连续2秒静音
                        self._stop_asr()
                        if self.frontend_server:
                            self.frontend_server.send_status(asr=False)
                        
        except KeyboardInterrupt:
            print("\n正在停止...")
        finally:
            self.stop()
    
    def _push_video_to_frontend(self):
        """持续推送视频帧到前端（独立于远程处理）"""
        last_push_time = 0
        push_interval = 0.1  # 100ms推送一次，约10fps
        
        while self.running:
            current_time = time.time()
            
            if current_time - last_push_time >= push_interval:
                last_push_time = current_time
                
                if self.last_frame is not None and self.frontend_server:
                    try:
                        _, buffer = cv2.imencode('.jpg', self.last_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        image_b64 = base64.b64encode(buffer).decode('utf-8')
                        self.frontend_server.send_frame(image_b64)
                    except:
                        pass
            
            time.sleep(0.03)
    
    def stop(self):
        """停止"""
        if self.streaming_active:
            self._stop_streaming_process()
        self.remote_client.disconnect()
        super().stop()
    
    def stop_streaming(self):
        """手动停止流式处理"""
        self._stop_streaming_process()


def main():
    agent = ExtendedAgent()
    
    try:
        agent.start()
    except KeyboardInterrupt:
        print("\n正在停止...")
        agent.stop()


if __name__ == "__main__":
    main()
