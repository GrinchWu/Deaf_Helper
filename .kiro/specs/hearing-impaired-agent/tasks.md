# Implementation Plan: Hearing Impaired Agent

## Overview

本实现计划将听障辅助Agent系统分解为可执行的编码任务。采用增量开发方式，从核心数据模型开始，逐步构建各个组件，最后整合为完整系统。使用Python作为实现语言，Hypothesis作为属性测试框架。

## Tasks

- [x] 1. 项目初始化和核心数据模型
  - [x] 1.1 创建项目结构和依赖配置
    - 创建 `src/` 目录结构
    - 创建 `pyproject.toml` 或 `requirements.txt`，包含依赖：opencv-python, httpx, hypothesis, pytest
    - _Requirements: 全局_
  - [x] 1.2 实现核心数据模型和枚举类型
    - 创建 `src/models.py`
    - 实现 FrameData, ProcessorConfig, VLLMConfig, MemoryConfig 数据类
    - 实现 UserLocation, UserState, ServiceType 枚举
    - 实现 SceneAnalysisResult, ServiceDecision, ContextEntry 数据类
    - 添加 to_dict() 和 from_dict() 方法用于JSON序列化
    - _Requirements: 3.5, 5.6_
  - [ ]* 1.3 编写数据模型属性测试
    - **Property 8: SceneAnalysisResult JSON Round-Trip**
    - **Property 16: Context JSON Round-Trip**
    - **Validates: Requirements 3.5, 5.6**

- [x] 2. Memory Manager 实现
  - [x] 2.1 实现 MemoryManager 类
    - 创建 `src/memory_manager.py`
    - 实现滑动窗口机制（使用 collections.deque）
    - 实现 store_context, get_recent_context 方法
    - 实现 get_memory_usage_mb, purge_old_context 方法
    - 实现 serialize, deserialize 方法
    - _Requirements: 5.1, 5.2, 5.3, 5.6_
  - [x] 2.2 编写 MemoryManager 属性测试

    - **Property 12: Sliding Window Memory Limit**
    - **Validates: Requirements 5.1, 5.2**
  - [ ]* 2.3 编写 MemoryManager 单元测试
    - 测试空缓冲区操作
    - 测试满缓冲区自动清理
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 3. Checkpoint - 数据层验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 4. VLLM Client 实现
  - [x] 4.1 实现 VLLMClient 类
    - 创建 `src/vllm_client.py`
    - 使用 httpx 实现异步HTTP客户端
    - 实现流式API调用（stream=True）
    - 实现指数退避重试逻辑（最多3次）
    - 配置 DashScope API endpoint 和 Qwen3-Omni-flash 模型
    - _Requirements: 2.1, 2.2, 2.3, 2.6_
  - [ ]* 4.2 编写 VLLMClient 属性测试
    - **Property 4: Stream Mode Always Enabled**
    - **Property 6: Retry with Exponential Backoff**
    - **Validates: Requirements 2.1, 2.3, 2.6**
  - [ ]* 4.3 编写 VLLMClient 单元测试
    - 测试请求格式正确性
    - 测试流式响应解析
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 5. Scene Analyzer 实现
  - [x] 5.1 实现 SceneAnalyzer 类
    - 创建 `src/scene_analyzer.py`
    - 实现 build_analysis_prompt 方法，包含场景描述、物体、位置、状态的请求
    - 实现 analyze_frame 方法，调用 VLLMClient 并解析响应
    - 实现响应解析逻辑，提取结构化信息
    - _Requirements: 2.4, 3.1, 3.2, 3.3, 3.4, 3.5_
  - [ ]* 5.2 编写 SceneAnalyzer 属性测试
    - **Property 5: Analysis Prompt Completeness**
    - **Property 7: SceneAnalysisResult Structure Completeness**
    - **Validates: Requirements 2.4, 3.1, 3.2, 3.3, 3.4**
  - [ ]* 5.3 编写 SceneAnalyzer 单元测试
    - 测试提示词生成
    - 测试枚举值映射
    - _Requirements: 3.3, 3.4_

- [x] 6. Service Decision Engine 实现
  - [x] 6.1 实现 ServiceDecisionEngine 类
    - 创建 `src/service_decision.py`
    - 实现 _is_social_scenario 方法（会议室、酒吧、餐厅、交谈状态）
    - 实现 _is_traffic_scenario 方法（街道 + 行走/骑行/驾驶）
    - 实现 decide 方法，返回 ServiceDecision
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [ ]* 6.2 编写 ServiceDecisionEngine 属性测试
    - **Property 9: ServiceDecision Completeness**
    - **Property 10: Service Type Present When Needed**
    - **Property 11: Scenario-Based Service Recommendation**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 4.6**
  - [ ]* 6.3 编写 ServiceDecisionEngine 单元测试
    - 测试社交场景识别
    - 测试交通场景识别
    - 测试未知场景处理
    - _Requirements: 4.5, 4.6_

- [x] 7. Checkpoint - 核心逻辑验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 8. Video Stream Processor 实现
  - [x] 8.1 实现 VideoStreamProcessor 类
    - 创建 `src/video_processor.py`
    - 使用 OpenCV 实现视频读取
    - 实现 process_video 异步生成器，逐帧处理
    - 实现帧采样逻辑（sample_rate 配置）
    - 实现损坏帧跳过逻辑
    - 实现帧数据 Base64 编码
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.4_
  - [ ]* 8.2 编写 VideoStreamProcessor 属性测试
    - **Property 1: Frame Sequential Processing**
    - **Property 2: Frame Extraction Count**
    - **Property 3: Corrupted Frame Resilience**
    - **Property 14: Configurable Sampling Rate Effect**
    - **Validates: Requirements 1.1, 1.2, 1.4, 5.4**
  - [ ]* 8.3 编写 VideoStreamProcessor 单元测试
    - 测试空视频处理
    - 测试单帧视频处理
    - _Requirements: 1.1, 1.2_

- [x] 9. Main Agent 整合
  - [x] 9.1 实现 HearingImpairedAgent 主类
    - 创建 `src/agent.py`
    - 整合所有组件
    - 实现 run 异步生成器方法
    - 实现自适应采样率调整逻辑
    - _Requirements: 5.5, 6.1, 6.2, 6.3, 6.4_
  - [ ]* 9.2 编写 Agent 属性测试
    - **Property 15: Adaptive Sampling on Low Activity**
    - **Validates: Requirements 5.5**
  - [ ]* 9.3 编写 Agent 集成测试
    - 测试完整处理流程
    - 测试输出格式
    - _Requirements: 全局_

- [x] 10. 入口脚本和配置
  - [x] 10.1 创建命令行入口
    - 创建 `src/main.py`
    - 实现命令行参数解析（视频路径、配置选项）
    - 实现配置加载
    - 实现结果输出（JSON格式）
    - _Requirements: 全局_
  - [x] 10.2 创建示例配置文件
    - 创建 `config.example.json`
    - 包含所有可配置参数的默认值
    - _Requirements: 全局_

- [x] 11. Final Checkpoint - 完整系统验证
  - 确保所有测试通过
  - 验证端到端流程
  - 如有问题请询问用户

## Notes

- 任务标记 `*` 的为可选任务，可跳过以加快MVP开发
- 每个任务引用具体需求以确保可追溯性
- Checkpoint任务用于增量验证
- 属性测试验证普遍正确性属性
- 单元测试验证特定示例和边界情况
