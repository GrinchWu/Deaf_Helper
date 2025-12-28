"""
前端WebSocket服务器 - 向浏览器推送实时数据

功能:
1. 接收来自ExtendedAgent的数据
2. 通过WebSocket推送到前端浏览器
3. 支持多个浏览器客户端连接

使用方法:
1. 启动此服务: python ws_frontend_server.py
2. 打开浏览器访问 front.html
3. 运行 ws_client.py (会自动连接此服务)
"""

import asyncio
import json
import base64
import threading
import time
from typing import Set, Optional, Dict, Any
from dataclasses import dataclass, asdict
from collections import deque

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("请安装: pip install websockets")
    exit(1)


@dataclass
class FrontendMessage:
    """发送到前端的消息"""
    type: str  # frame, detection, transcript, tom, status
    data: Dict[str, Any]


class FrontendServer:
    """前端WebSocket服务器"""
    
    def __init__(self, host: str = "localhost", port: int = 8766):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server = None
        self.loop = None
        self.running = False
        
        # 数据队列
        self.message_queue: deque = deque(maxlen=100)
        
        # 最新状态缓存
        self.latest_frame: Optional[str] = None
        self.latest_detection: Optional[Dict] = None
        self.latest_transcripts: deque = deque(maxlen=20)
        
    async def handler(self, websocket: WebSocketServerProtocol):
        """处理WebSocket连接"""
        self.clients.add(websocket)
        client_id = id(websocket)
        print(f"[前端服务] 客户端连接: {client_id}, 当前连接数: {len(self.clients)}")
        
        try:
            # 发送当前状态
            if self.latest_frame:
                await websocket.send(json.dumps({
                    "type": "frame",
                    "image": self.latest_frame
                }))
            
            if self.latest_detection:
                await websocket.send(json.dumps({
                    "type": "detection",
                    **self.latest_detection
                }))
            
            # 发送历史转录
            for t in self.latest_transcripts:
                await websocket.send(json.dumps({
                    "type": "transcript",
                    **t
                }))
            
            # 保持连接
            async for message in websocket:
                # 处理来自前端的消息（如果需要）
                try:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                except:
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[前端服务] 客户端断开: {client_id}, 当前连接数: {len(self.clients)}")
    
    async def handle_client_message(self, websocket: WebSocketServerProtocol, data: Dict):
        """处理来自前端的消息"""
        cmd = data.get("command")
        if cmd == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
    
    async def broadcast(self, message: Dict):
        """广播消息到所有客户端"""
        if not self.clients:
            return
        
        msg_str = json.dumps(message, ensure_ascii=False)
        
        # 并发发送
        tasks = [client.send(msg_str) for client in self.clients.copy()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def send_frame(self, image_b64: str):
        """发送视频帧"""
        self.latest_frame = image_b64
        if self.loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "frame", "image": image_b64}),
                self.loop
            )
    
    def send_detection(self, persons: list, speaker_id: Optional[str] = None):
        """发送检测结果"""
        data = {
            "persons": persons,
            "speaker_id": speaker_id
        }
        self.latest_detection = data
        
        if self.loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "detection", **data}),
                self.loop
            )
    
    def send_transcript(self, speaker: str, text: str):
        """发送转录结果"""
        data = {"speaker": speaker, "text": text}
        self.latest_transcripts.append(data)
        
        if self.loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "transcript", **data}),
                self.loop
            )
    
    def send_tom(self, emotion: str, confidence: float, intent: str, 
                 mental_state: str, suggested_response: str):
        """发送ToM分析结果"""
        data = {
            "emotion": emotion,
            "confidence": confidence,
            "intent": intent,
            "mental_state": mental_state,
            "suggested_response": suggested_response
        }
        
        print(f"[前端服务] send_tom被调用: emotion={emotion}, clients={len(self.clients)}")
        
        if self.loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "tom", **data}),
                self.loop
            )
        else:
            print(f"[前端服务] 警告: loop={self.loop is not None}, running={self.running}")
    
    def send_status(self, remote: Optional[bool] = None, asr: Optional[bool] = None):
        """发送系统状态"""
        data = {}
        if remote is not None:
            data["remote"] = remote
        if asr is not None:
            data["asr"] = asr
        
        if data and self.loop and self.running:
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "status", **data}),
                self.loop
            )
    
    async def _run_server(self):
        """运行服务器"""
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port,
            max_size=10*1024*1024  # 10MB
        )
        print(f"[前端服务] WebSocket服务器启动: ws://{self.host}:{self.port}")
        await self.server.wait_closed()
    
    def start(self):
        """在独立线程中启动服务器"""
        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.running = True
            self.loop.run_until_complete(self._run_server())
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(0.5)  # 等待服务器启动
        return thread
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.server:
            self.server.close()


# 全局实例
_frontend_server: Optional[FrontendServer] = None


def get_frontend_server() -> FrontendServer:
    """获取前端服务器单例"""
    global _frontend_server
    if _frontend_server is None:
        _frontend_server = FrontendServer()
        _frontend_server.start()
    return _frontend_server


if __name__ == "__main__":
    # 独立运行测试
    server = FrontendServer()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server.running = True
        server.loop = loop
        
        print("前端WebSocket服务器启动中...")
        loop.run_until_complete(server._run_server())
    except KeyboardInterrupt:
        print("\n停止服务器...")
        server.stop()
