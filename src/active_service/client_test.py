"""
本地客户端测试 - 采集摄像头和麦克风数据，发送到远程服务器

使用前:
1. 修改 SERVER_URL 为你的服务器地址
2. 安装依赖: pip install websockets opencv-python pyaudio numpy

运行: python client_test.py
"""

import asyncio
import json
import base64
import time
import threading
import queue
import cv2
import numpy as np

try:
    import pyaudio
except ImportError:
    print("请安装: pip install pyaudio")
    pyaudio = None

try:
    import websockets
except ImportError:
    print("请安装: pip install websockets")
    exit(1)


# ==================== 配置 ====================
# 修改为你的服务器地址（SSH端口转发后的地址）
# 方式1: 直接连接（如果服务器端口对外开放）
# SERVER_URL = "ws://connect.nmb1.seetacloud.com:8765"

# 方式2: 通过SSH端口转发（推荐）
# 先运行: ssh -p 31801 -L 8765:localhost:8765 root@connect.nmb1.seetacloud.com
# 然后使用本地地址:
SERVER_URL = "ws://localhost:8765"

SAMPLE_RATE = 16000
AUDIO_CHUNK = 3200  # 200ms


# ==================== 数据采集 ====================
class DataCapture:
    """采集摄像头和麦克风数据"""
    
    def __init__(self):
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.audio_queue = queue.Queue(maxsize=10)
        
        self.cap = None
        self.mic = None
        self.audio_stream = None
    
    def start(self):
        """启动采集"""
        self.running = True
        
        # 启动视频采集
        threading.Thread(target=self._capture_video, daemon=True).start()
        
        # 启动音频采集
        if pyaudio:
            threading.Thread(target=self._capture_audio, daemon=True).start()
        
        print("数据采集已启动")
    
    def stop(self):
        """停止采集"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.mic:
            self.mic.terminate()
    
    def _capture_video(self):
        """视频采集线程"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
            time.sleep(0.1)  # 10fps
    
    def _capture_audio(self):
        """音频采集线程"""
        try:
            self.mic = pyaudio.PyAudio()
            self.audio_stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK
            )
            
            while self.running:
                try:
                    data = self.audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                    try:
                        self.audio_queue.put_nowait(data)
                    except queue.Full:
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(data)
                        except:
                            pass
                except:
                    time.sleep(0.1)
        except Exception as e:
            print(f"音频采集错误: {e}")
    
    def get_frame(self) -> bytes:
        """获取图像（base64编码）"""
        try:
            frame = self.frame_queue.get_nowait()
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            return base64.b64encode(buffer).decode('utf-8')
        except queue.Empty:
            return ""
    
    def get_audio(self) -> bytes:
        """获取音频（base64编码）"""
        try:
            audio = self.audio_queue.get_nowait()
            return base64.b64encode(audio).decode('utf-8')
        except queue.Empty:
            return ""


# ==================== WebSocket客户端 ====================
async def test_connection():
    """测试WebSocket连接"""
    print(f"\n连接服务器: {SERVER_URL}")
    
    capture = DataCapture()
    capture.start()
    
    # 等待采集启动
    time.sleep(1)
    
    try:
        async with websockets.connect(SERVER_URL) as ws:
            print("连接成功！\n")
            
            for i in range(5):  # 发送5次测试数据
                # 获取数据
                image_b64 = capture.get_frame()
                audio_b64 = capture.get_audio()
                
                # 构建请求
                request = {
                    "data": {
                        "text": f"测试消息 #{i+1}",
                        "image": image_b64,
                        "audio": audio_b64
                    }
                }
                
                # 发送
                print(f"[发送 #{i+1}] 图像: {len(image_b64)//1024}KB, 音频: {len(audio_b64)//1024}KB")
                await ws.send(json.dumps(request, ensure_ascii=False))
                
                # 接收响应
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                result = json.loads(response)
                
                print(f"[响应 #{i+1}] {result.get('message', 'OK')}")
                print(f"         服务器时间: {result.get('server_time', 'N/A')}")
                print(f"         回显: {result.get('echo_text', 'N/A')}")
                print()
                
                await asyncio.sleep(1)
            
            print("测试完成！")
            
    except ConnectionRefusedError:
        print("连接被拒绝，请检查:")
        print("1. 服务器是否已启动")
        print("2. 端口是否正确")
        print("3. 是否需要SSH端口转发")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        capture.stop()


def main():
    print("=" * 50)
    print("WebSocket客户端测试")
    print("=" * 50)
    print(f"服务器地址: {SERVER_URL}")
    print()
    print("使用SSH端口转发连接远程服务器:")
    print("  ssh -p 31801 -L 8765:localhost:8765 root@connect.nmb1.seetacloud.com")
    print()
    
    asyncio.run(test_connection())


if __name__ == "__main__":
    main()
