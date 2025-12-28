"""
远程服务器端 - WebSocket服务 + YOLOv5人物检测
部署到服务器: ssh -p 31801 root@connect.nmb1.seetacloud.com

使用方法:
1. 上传此文件到服务器
2. 确保yolov5目录存在且有yolov5n.pt权重
3. 运行: conda activate ezvsl && python ws_server.py --port 8765 --yolo-path /path/to/yolov5
"""

import asyncio
import json
import base64
import argparse
import sys
import os
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from websockets.server import serve
except ImportError:
    print("请安装: pip install websockets")
    sys.exit(1)

import numpy as np
import cv2

# YOLOv5路径（根据实际情况修改）
YOLOV5_PATH = os.getenv("YOLOV5_PATH", "/root/yolov5")


class YOLOv5Detector:
    """YOLOv5人物检测器"""
    
    def __init__(self, yolo_path: str = YOLOV5_PATH, weights: str = "yolov5n.pt"):
        self.yolo_path = yolo_path
        self.weights = weights
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载YOLOv5模型"""
        try:
            # 添加yolov5到路径
            if self.yolo_path not in sys.path:
                sys.path.insert(0, self.yolo_path)
            
            import torch
            
            # 加载模型
            weights_path = os.path.join(self.yolo_path, self.weights)
            if not os.path.exists(weights_path):
                weights_path = self.weights  # 尝试直接使用权重名
            
            self.model = torch.hub.load(self.yolo_path, 'custom', path=weights_path, source='local')
            self.model.classes = [0]  # 只检测person类（COCO类别0）
            self.model.conf = 0.5  # 置信度阈值
            
            print(f"YOLOv5模型加载成功: {weights_path}")
        except Exception as e:
            print(f"YOLOv5模型加载失败: {e}")
            print("将使用备用方案")
            self.model = None
    
    def detect_persons(self, image_base64: str) -> dict:
        """
        检测图像中的人物
        
        Args:
            image_base64: base64编码的图像
        
        Returns:
            {
                "persons": [
                    {"label": "person1", "bbox": [x1, y1, x2, y2], "confidence": 0.95},
                    {"label": "person2", "bbox": [x1, y1, x2, y2], "confidence": 0.88},
                    ...
                ],
                "count": 2
            }
        """
        try:
            # 解码图像
            img_bytes = base64.b64decode(image_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"persons": [], "count": 0, "error": "图像解码失败"}
            
            if self.model is None:
                return {"persons": [], "count": 0, "error": "模型未加载"}
            
            # YOLOv5推理
            results = self.model(img)
            
            # 解析结果
            detections = results.pandas().xyxy[0]  # 获取DataFrame格式结果
            
            # 筛选person类别
            persons = detections[detections['name'] == 'person']
            
            # 按x坐标从左到右排序
            persons = persons.sort_values(by='xmin')
            
            # 构建输出
            person_list = []
            for idx, (_, row) in enumerate(persons.iterrows(), 1):
                person_list.append({
                    "label": f"person{idx}",
                    "bbox": [
                        int(row['xmin']),
                        int(row['ymin']),
                        int(row['xmax']),
                        int(row['ymax'])
                    ],
                    "confidence": round(float(row['confidence']), 3)
                })
            
            return {
                "persons": person_list,
                "count": len(person_list)
            }
            
        except Exception as e:
            return {"persons": [], "count": 0, "error": str(e)}


class WebSocketServer:
    """WebSocket服务端"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, yolo_path: str = YOLOV5_PATH):
        self.host = host
        self.port = port
        self.detector = YOLOv5Detector(yolo_path=yolo_path)
    
    async def handler(self, websocket):
        """处理客户端连接"""
        client = websocket.remote_address
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 客户端连接: {client}")
        
        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    cmd = request.get("command", "")
                    data = request.get("data", {})
                    
                    print(f"[收到] 命令: {cmd}")
                    
                    if cmd == "detect_person":
                        # 人物检测
                        image_b64 = data.get("image", "")
                        if image_b64:
                            result = self.detector.detect_persons(image_b64)
                            result["status"] = "success"
                            result["timestamp"] = datetime.now().isoformat()
                        else:
                            result = {"status": "error", "message": "无图像数据"}
                    
                    elif cmd == "ping":
                        result = {"status": "success", "message": "pong"}
                    
                    else:
                        result = {"status": "error", "message": f"未知命令: {cmd}"}
                    
                    await websocket.send(json.dumps(result, ensure_ascii=False))
                    
                    if "persons" in result:
                        print(f"[响应] 检测到 {result.get('count', 0)} 人")
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"status": "error", "message": "JSON解析失败"}))
                except Exception as e:
                    await websocket.send(json.dumps({"status": "error", "message": str(e)}))
                    
        except Exception as e:
            print(f"[断开] {client}: {e}")
    
    async def start(self):
        """启动服务"""
        print("=" * 50)
        print("WebSocket服务器 + YOLOv5人物检测")
        print(f"监听: ws://{self.host}:{self.port}")
        print("=" * 50)
        
        async with serve(self.handler, self.host, self.port):
            await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="WebSocket服务器 + YOLOv5")
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
