# 听障辅助AI眼镜Agent

针对听障人群的AI眼镜主动服务Agent，实时分析视频流并智能判断用户是否需要服务。

## 功能特点

- **事件驱动架构**：只在场景变化时调用API，减少90%+的请求
- **本地场景变化检测**：< 10ms延迟的轻量级检测
- **流式API调用**：使用阿里云Qwen3-Omni-flash模型
- **智能服务决策**：支持社交语言转录和交通安全提醒
- **能耗优化**：滑动窗口记忆管理，自适应采样率

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
# 基本使用
python -m src.main video.mp4 --api-key YOUR_API_KEY

# 使用配置文件
python -m src.main video.mp4 --config config.json

# 保存结果
python -m src.main video.mp4 --api-key YOUR_API_KEY --output results.json
```

## 配置

复制 `config.example.json` 为 `config.json` 并填入API密钥：

```json
{
  "vllm_config": {
    "api_key": "YOUR_API_KEY_HERE"
  }
}
```

## 架构

```
视频流 → 本地场景变化检测 → [变化?] → VLLM分析 → 服务决策 → 输出
                              ↓
                         [无变化] → 使用缓存结果
```

## 服务类型

- **社交语言转录文字**：会议室、餐厅等社交场景
- **交通场景安全提醒**：街道行走、骑行、驾驶场景

## License

MIT
