"""
远程服务器端 - WebSocket服务 + YOLOv5人物检测 + EZ-VSL音源定位

功能:
1. 接收客户端的图像流（每0.5秒一帧）
2. YOLOv5检测人物，按从左到右排序标记为person1, person2...
3. 调用EZ-VSL进行音源定位，确定说话人
4. 返回检测结果和speaker_id

部署到服务器:
1. 上传此文件到 /root/autodl-tmp/ws_server.py
2. conda activate ezvsl
3. python ws_server.py --port 8765
"""

import asyncio
import json
import base64
import argparse
import sys
import os
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from websockets.server import serve
except ImportError:
    print("请安装: pip install websockets")
    sys.exit(1)

import numpy as np
import cv2

# ==================== 配置 ====================
YOLOV5_PATH = os.getenv("YOLOV5_PATH", "/root/yolov5")
EZVSL_PATH = "/root/autodl-tmp/EZ-VSL"
STEP_DURATION = 0.5  # 每个step的时长（秒）
OUTPUT_BASE_DIR = "/root/autodl-tmp/streaming_sessions"


# ==================== YOLOv5检测器 ====================
class YOLOv5Detector:
    """YOLOv5人物检测器 - 支持流式处理"""
    
    def __init__(self, yolo_path: str = YOLOV5_PATH, weights: str = "yolov5n.pt"):
        self.yolo_path = yolo_path
        self.weights = weights
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载YOLOv5模型"""
        try:
            if self.yolo_path not in sys.path:
                sys.path.insert(0, self.yolo_path)
            
            import torch
            
            weights_path = os.path.join(self.yolo_path, self.weights)
            if not os.path.exists(weights_path):
                # 尝试在根目录查找
                weights_path = f"/root/{self.weights}"
            
            self.model = torch.hub.load(self.yolo_path, 'custom', path=weights_path, source='local')
            self.model.classes = [0]  # 只检测person类
            self.model.conf = 0.25  # 降低置信度阈值以检测更多人
            
            print(f"YOLOv5模型加载成功: {weights_path}")
        except Exception as e:
            print(f"YOLOv5模型加载失败: {e}")
            self.model = None
    
    def detect_persons(self, image: np.ndarray, step: int = 0) -> Dict:
        """
        检测图像中的人物
        
        Args:
            image: 图像数组
            step: 当前step编号
        
        Returns:
            检测结果字典
        """
        timestamp = step * STEP_DURATION
        
        if image is None:
            return {"step": step, "timestamp": timestamp, "persons": [], "count": 0, "error": "图像为空"}
        
        if self.model is None:
            return {"step": step, "timestamp": timestamp, "persons": [], "count": 0, "error": "模型未加载"}
        
        try:
            # YOLOv5推理
            results = self.model(image)
            
            # 解析结果
            detections = results.pandas().xyxy[0]
            
            # 筛选person类别
            persons = detections[detections['name'] == 'person']
            
            # 按x坐标从左到右排序
            persons = persons.sort_values(by='xmin')
            
            # 获取图像尺寸用于归一化
            h, w = image.shape[:2]
            
            # 构建输出
            person_list = []
            for idx, (_, row) in enumerate(persons.iterrows(), 1):
                # 原始像素坐标
                bbox_pixel = [
                    int(row['xmin']),
                    int(row['ymin']),
                    int(row['xmax']),
                    int(row['ymax'])
                ]
                
                # 归一化坐标 (用于EZ-VSL)
                bbox_normalized = [
                    round(row['xmin'] / w, 4),
                    round(row['ymin'] / h, 4),
                    round(row['xmax'] / w, 4),
                    round(row['ymax'] / h, 4)
                ]
                
                person_list.append({
                    "label": f"person{idx}",
                    "bbox": bbox_pixel,
                    "bbox_normalized": bbox_normalized,
                    "confidence": round(float(row['confidence']), 4)
                })
            
            return {
                "step": step,
                "timestamp": timestamp,
                "persons": person_list,
                "count": len(person_list),
                "image_size": [w, h]
            }
            
        except Exception as e:
            return {"step": step, "timestamp": timestamp, "persons": [], "count": 0, "error": str(e)}


# ==================== 会话管理器 ====================
class StreamingSession:
    """流式处理会话"""
    
    def __init__(self, session_id: str, output_dir: str):
        self.session_id = session_id
        self.output_dir = output_dir
        self.current_step = 0
        self.results: List[Dict] = []
        self.speaker_results: List[Dict] = []  # EZ-VSL结果
        
        self.frames_dir = os.path.join(output_dir, "frames")
        self.yolo_dir = os.path.join(output_dir, "yolo_split")
        self.audio_dir = os.path.join(output_dir, "audio")
        self.attribution_dir = os.path.join(output_dir, "attribution")
        
        # 创建目录
        for d in [self.frames_dir, self.yolo_dir, self.audio_dir, self.attribution_dir]:
            os.makedirs(d, exist_ok=True)
        
        # 音频缓冲
        self.audio_chunks: List[bytes] = []
        
        # SPEAKER绑定状态
        self.speaker_to_person: Dict[str, Optional[str]] = {}
        self.assigned_persons: set = set()
        
        print(f"[Session {session_id}] 创建于 {output_dir}")
    
    def save_frame(self, step: int, image: np.ndarray):
        """保存帧图像"""
        filepath = os.path.join(self.frames_dir, f"step_{step:03d}.jpg")
        cv2.imwrite(filepath, image)
        return filepath
    
    def save_yolo_result(self, result: Dict):
        """保存YOLO检测结果"""
        step = result["step"]
        
        # 保存单个step的JSON
        filepath = os.path.join(self.yolo_dir, f"step_{step:03d}_yolo.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.results.append(result)
    
    def save_audio_chunk(self, step: int, audio_data: bytes):
        """保存音频数据块"""
        self.audio_chunks.append(audio_data)
        
        # 保存单个音频块
        filepath = os.path.join(self.audio_dir, f"step_{step:03d}.raw")
        with open(filepath, 'wb') as f:
            f.write(audio_data)
    
    def save_combined_audio(self):
        """保存合并的音频文件"""
        if not self.audio_chunks:
            return None
        
        audio_path = os.path.join(self.audio_dir, "audio.wav")
        
        # 合并音频数据
        combined = b''.join(self.audio_chunks)
        
        # 写入WAV文件
        with wave.open(audio_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(16000)
            wav.writeframes(combined)
        
        return audio_path
    
    def save_speaker_result(self, step: int, speaker_id: Optional[str], yolo_result: Dict):
        """保存说话人归属结果"""
        timestamp = step * STEP_DURATION
        
        # 构建输出
        speakers = []
        if speaker_id and speaker_id != "OUT_OF_FRAME":
            # 找到对应的person
            for p in yolo_result.get("persons", []):
                if p["label"] == speaker_id:
                    speakers.append({
                        "person_id": speaker_id,
                        "bbox": p.get("bbox_normalized", p["bbox"]),
                        "confidence": p["confidence"]
                    })
                    break
        elif speaker_id == "OUT_OF_FRAME":
            speakers.append({
                "person_id": "OUT_OF_FRAME",
                "bbox": [-1, -1, -1, -1],
                "confidence": 0.0
            })
        
        output_data = {
            "step": step,
            "timestamp": timestamp,
            "speakers": speakers,
            "speaker_id": speaker_id
        }
        
        # 保存到文件
        filepath = os.path.join(self.attribution_dir, f"output_step_{step}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.speaker_results.append(output_data)
        return output_data
    
    def get_summary(self) -> Dict:
        """获取会话汇总"""
        return {
            "session_id": self.session_id,
            "total_steps": self.current_step,
            "output_dir": self.output_dir,
            "frames": self.results,
            "speaker_results": self.speaker_results
        }
    
    def save_summary(self):
        """保存汇总文件"""
        # 保存合并的YOLO JSON（用于EZ-VSL）
        yolo_combined = {
            "video_info": {
                "session_id": self.session_id,
                "total_steps": self.current_step,
                "step_duration": STEP_DURATION
            },
            "frames": [
                {
                    "step": r["step"],
                    "frame": int(r["timestamp"] * 30),
                    "persons": [
                        {
                            "label": p["label"],
                            "bbox": p.get("bbox_normalized", p["bbox"]),
                            "confidence": p["confidence"]
                        }
                        for p in r.get("persons", [])
                    ],
                    "count": r.get("count", 0)
                }
                for r in self.results
            ]
        }
        
        yolo_filepath = os.path.join(self.output_dir, "persons.json")
        with open(yolo_filepath, 'w', encoding='utf-8') as f:
            json.dump(yolo_combined, f, indent=2, ensure_ascii=False)
        
        # 保存总汇总
        summary = self.get_summary()
        summary_filepath = os.path.join(self.output_dir, "summary.json")
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"[Session {self.session_id}] 汇总已保存")


# ==================== EZ-VSL处理器 ====================
class EZVSLProcessor:
    """EZ-VSL音源定位处理器"""
    
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path or os.path.join(EZVSL_PATH, "vggsound_10k/best.pth")
        self.inferencer = None
        self._load_model()
    
    def _load_model(self):
        """加载EZ-VSL模型"""
        try:
            sys.path.insert(0, EZVSL_PATH)
            from diart_multi_source.ezvsl_inference import EZVSLInference
            
            if os.path.exists(self.checkpoint_path):
                self.inferencer = EZVSLInference(self.checkpoint_path)
                print(f"EZ-VSL模型加载成功: {self.checkpoint_path}")
            else:
                print(f"EZ-VSL模型文件不存在: {self.checkpoint_path}")
                self.inferencer = None
        except Exception as e:
            print(f"EZ-VSL模型加载失败: {e}")
            self.inferencer = None
    
    def process_frame(self, frame_path: str, audio_path: str, 
                      audio_start: float, audio_duration: float = 3.0) -> Dict:
        """
        处理单帧，获取热力图锚点
        
        Returns:
            {"anchor": {"x": 0.5, "y": 0.5}, "peak_energy": 0.8, "is_valid": True}
        """
        if self.inferencer is None:
            return {"anchor": {"x": 0.5, "y": 0.5}, "peak_energy": 0.0, "is_valid": False}
        
        try:
            result = self.inferencer.process_frame(
                frame_path, audio_path,
                audio_start=audio_start,
                audio_duration=audio_duration
            )
            return result
        except Exception as e:
            print(f"EZ-VSL处理失败: {e}")
            return {"anchor": {"x": 0.5, "y": 0.5}, "peak_energy": 0.0, "is_valid": False}
    
    def match_speaker_to_person(self, anchor: Dict, persons: List[Dict], 
                                 tolerance: float = 0.15) -> Optional[str]:
        """
        将热力图锚点匹配到人物（改进版：使用距离匹配）
        
        Args:
            anchor: {"x": 0.5, "y": 0.5} 归一化坐标
            persons: [{"label": "person1", "bbox_normalized": [x1,y1,x2,y2], ...}]
            tolerance: 容差范围（归一化坐标）
        
        Returns:
            匹配的person_id 或 "OUT_OF_FRAME" 或 None
        """
        if not persons:
            return "OUT_OF_FRAME"
        
        x, y = anchor.get("x", 0.5), anchor.get("y", 0.5)
        
        best_match = None
        best_distance = float('inf')
        
        for person in persons:
            bbox = person.get("bbox_normalized", person.get("bbox", [0, 0, 1, 1]))
            
            if len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                
                # 计算bbox中心
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                # 计算锚点到bbox中心的距离
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                
                # 扩展bbox范围（加上容差）
                x_min_ext = x_min - tolerance
                x_max_ext = x_max + tolerance
                y_min_ext = y_min - tolerance
                y_max_ext = y_max + tolerance
                
                # 检查锚点是否在扩展bbox内
                if x_min_ext <= x <= x_max_ext and y_min_ext <= y <= y_max_ext:
                    if distance < best_distance:
                        best_distance = distance
                        best_match = person["label"]
        
        # 如果没有匹配，选择最近的人（如果距离在合理范围内）
        if best_match is None:
            for person in persons:
                bbox = person.get("bbox_normalized", person.get("bbox", [0, 0, 1, 1]))
                if len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    
                    # 如果距离小于0.35（归一化坐标），认为可能是这个人
                    if distance < 0.35 and distance < best_distance:
                        best_distance = distance
                        best_match = person["label"]
        
        return best_match if best_match else "OUT_OF_FRAME"


# ==================== WebSocket服务器 ====================
class WebSocketServer:
    """WebSocket服务端 - 支持流式处理"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, 
                 yolo_path: str = YOLOV5_PATH):
        self.host = host
        self.port = port
        self.detector = YOLOv5Detector(yolo_path=yolo_path)
        self.ezvsl = EZVSLProcessor()
        self.sessions: Dict[str, StreamingSession] = {}
    
    def _get_or_create_session(self, client_id: str) -> StreamingSession:
        """获取或创建会话"""
        if client_id not in self.sessions:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"{timestamp}_{client_id[:8]}"
            output_dir = os.path.join(OUTPUT_BASE_DIR, session_id)
            self.sessions[client_id] = StreamingSession(session_id, output_dir)
        return self.sessions[client_id]
    
    async def handler(self, websocket):
        """处理客户端连接"""
        client = str(websocket.remote_address)
        client_id = f"{client}_{int(time.time())}"
        session = None
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 客户端连接: {client}")
        
        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    cmd = request.get("command", "")
                    data = request.get("data", {})
                    
                    result = await self._handle_command(cmd, data, client_id)
                    await websocket.send(json.dumps(result, ensure_ascii=False))
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"status": "error", "message": "JSON解析失败"}))
                except Exception as e:
                    await websocket.send(json.dumps({"status": "error", "message": str(e)}))
                    
        except Exception as e:
            print(f"[断开] {client}: {e}")
        finally:
            # 清理会话
            if client_id in self.sessions:
                session = self.sessions[client_id]
                session.save_summary()
    
    async def _handle_command(self, cmd: str, data: Dict, client_id: str) -> Dict:
        """处理命令"""
        
        if cmd == "start_session":
            # 开始新会话
            session = self._get_or_create_session(client_id)
            session.current_step = 0
            return {
                "status": "success",
                "session_id": session.session_id,
                "message": "会话已开始"
            }
        
        elif cmd == "process_frame":
            # 处理单帧
            session = self._get_or_create_session(client_id)
            
            step = data.get("step", session.current_step + 1)
            session.current_step = step
            
            # 解码图像
            image_b64 = data.get("image", "")
            if not image_b64:
                return {"status": "error", "error": "无图像数据"}
            
            img_bytes = base64.b64decode(image_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"status": "error", "error": "图像解码失败"}
            
            # 保存帧
            frame_path = session.save_frame(step, image)
            
            # 处理音频（如果有）
            audio_b64 = data.get("audio")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                session.save_audio_chunk(step, audio_data)
            
            # YOLOv5检测
            yolo_result = self.detector.detect_persons(image, step)
            session.save_yolo_result(yolo_result)
            
            # EZ-VSL音源定位（如果有足够的音频数据）
            speaker_id = None
            if len(session.audio_chunks) >= 6:  # 至少3秒音频
                audio_path = session.save_combined_audio()
                if audio_path:
                    timestamp = step * STEP_DURATION
                    audio_start = max(0, timestamp - 1.5)
                    
                    ezvsl_result = self.ezvsl.process_frame(
                        frame_path, audio_path,
                        audio_start=audio_start,
                        audio_duration=3.0
                    )
                    
                    if ezvsl_result.get("is_valid"):
                        # 匹配说话人
                        persons_normalized = [
                            {
                                "label": p["label"],
                                "bbox_normalized": p.get("bbox_normalized", p["bbox"]),
                                "confidence": p["confidence"]
                            }
                            for p in yolo_result.get("persons", [])
                        ]
                        speaker_id = self.ezvsl.match_speaker_to_person(
                            ezvsl_result["anchor"],
                            persons_normalized
                        )
            
            # 保存说话人结果
            speaker_result = session.save_speaker_result(step, speaker_id, yolo_result)
            
            # 构建响应
            response = {
                "status": "success",
                "step": step,
                "timestamp": step * STEP_DURATION,
                "persons": yolo_result.get("persons", []),
                "count": yolo_result.get("count", 0),
                "speaker_id": speaker_id
            }
            
            print(f"[Step {step}] 检测到 {response['count']} 人, speaker={speaker_id}")
            return response
        
        elif cmd == "end_session":
            # 结束会话
            if client_id in self.sessions:
                session = self.sessions[client_id]
                session.save_combined_audio()
                session.save_summary()
                
                summary = session.get_summary()
                del self.sessions[client_id]
                
                return {
                    "status": "success",
                    "message": "会话已结束",
                    "summary": summary
                }
            return {"status": "error", "error": "会话不存在"}
        
        elif cmd == "detect_person":
            # 单次检测（兼容旧接口）
            image_b64 = data.get("image", "")
            if not image_b64:
                return {"status": "error", "message": "无图像数据"}
            
            img_bytes = base64.b64decode(image_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"status": "error", "message": "图像解码失败"}
            
            result = self.detector.detect_persons(image, step=0)
            result["status"] = "success"
            result["timestamp"] = datetime.now().isoformat()
            return result
        
        elif cmd == "ping":
            return {"status": "success", "message": "pong"}
        
        else:
            return {"status": "error", "message": f"未知命令: {cmd}"}
    
    async def start(self):
        """启动服务"""
        print("=" * 60)
        print("WebSocket服务器 + YOLOv5 + EZ-VSL")
        print(f"监听: ws://{self.host}:{self.port}")
        print(f"输出目录: {OUTPUT_BASE_DIR}")
        print("=" * 60)
        
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        async with serve(self.handler, self.host, self.port, max_size=10*1024*1024):
            await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="WebSocket服务器 + YOLOv5 + EZ-VSL")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--yolo-path", default=YOLOV5_PATH, help="YOLOv5目录路径")
    args = parser.parse_args()
    
    server = WebSocketServer(host=args.host, port=args.port, yolo_path=args.yolo_path)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == "__main__":
    main()
