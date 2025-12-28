"""
远程服务器端测试 - WebSocket服务
部署到服务器: ssh -p 31801 root@connect.nmb1.seetacloud.com

安装依赖: pip install websockets
启动命令: python server_test.py --port 8765

如需conda环境:
  conda activate your_env && python server_test.py --port 8765
"""

import asyncio
import json
import base64
import argparse
import subprocess
import os
from datetime import datetime

try:
    from websockets.server import serve
except ImportError:
    print("请安装: pip install websockets")
    exit(1)


# ==================== 配置 ====================
# 如果任务需要特定conda环境，设置环境名称
CONDA_ENV = None  # 例如: "myenv" 或 None表示不使用conda


class TestServer:
    """测试用WebSocket服务端"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765, conda_env: str = None):
        self.host = host
        self.port = port
        self.conda_env = conda_env
    
    def run_in_conda(self, python_code: str) -> str:
        """在指定conda环境中执行Python代码"""
        if not self.conda_env:
            # 直接执行
            result = subprocess.run(
                ["python", "-c", python_code],
                capture_output=True, text=True, timeout=30
            )
        else:
            # 在conda环境中执行
            # 方式1: 使用conda run
            result = subprocess.run(
                ["conda", "run", "-n", self.conda_env, "python", "-c", python_code],
                capture_output=True, text=True, timeout=30
            )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr}"
    
    def run_script_in_conda(self, script_path: str, args: list = None) -> str:
        """在指定conda环境中执行Python脚本"""
        cmd = ["python", script_path] + (args or [])
        
        if self.conda_env:
            cmd = ["conda", "run", "-n", self.conda_env] + cmd
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr}"
    
    async def process_task(self, data: dict) -> dict:
        """处理任务 - 可在此调用conda环境中的模型"""
        text = data.get("text", "")
        has_image = "image" in data
        has_audio = "audio" in data
        
        # 示例：在conda环境中执行简单任务
        if self.conda_env:
            # 调用conda环境中的Python
            code = f'print("处理文本: {text[:20]}")'
            output = self.run_in_conda(code)
        else:
            output = f"处理文本: {text[:20]}"
        
        return {
            "status": "success",
            "conda_env": self.conda_env or "default",
            "output": output,
            "received": {
                "text_length": len(text),
                "has_image": has_image,
                "has_audio": has_audio
            }
        }
    
    async def handler(self, websocket):
        """处理客户端连接"""
        client = websocket.remote_address
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 客户端连接: {client}")
        
        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    data = request.get("data", {})
                    
                    print(f"[收到] 文本: '{data.get('text', '')[:30]}...'")
                    
                    # 处理任务
                    result = await self.process_task(data)
                    result["server_time"] = datetime.now().isoformat()
                    
                    await websocket.send(json.dumps(result, ensure_ascii=False))
                    print(f"[响应] 已发送")
                    
                except Exception as e:
                    await websocket.send(json.dumps({"status": "error", "message": str(e)}))
                    
        except Exception as e:
            print(f"[断开] {client}: {e}")
    
    async def start(self):
        """启动服务"""
        print("=" * 50)
        print(f"WebSocket服务器")
        print(f"监听: ws://{self.host}:{self.port}")
        print(f"Conda环境: {self.conda_env or '默认'}")
        print("=" * 50)
        
        async with serve(self.handler, self.host, self.port):
            await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="WebSocket服务器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--conda", default=CONDA_ENV, help="Conda环境名称")
    args = parser.parse_args()
    
    server = TestServer(host=args.host, port=args.port, conda_env=args.conda)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n服务器已停止")


if __name__ == "__main__":
    main()
