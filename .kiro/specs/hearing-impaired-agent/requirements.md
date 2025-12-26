# Requirements Document

## Introduction

本功能是针对听障人群的AI眼镜Agent系统的核心模块，负责实时分析视频流并主动判断用户是否需要服务以及需要什么类型的服务。系统需要极速响应，支持长时间持续运行，并有效管理能耗和记忆存储。

## Glossary

- **Video_Stream_Processor**: 视频流处理器，负责逐帧读取和处理视频数据
- **Scene_Analyzer**: 场景分析器，调用VLLM模型分析画面内容
- **Service_Decision_Engine**: 服务决策引擎，判断是否需要提供服务及服务类型
- **Memory_Manager**: 记忆管理器，负责上下文存储和能耗优化
- **VLLM_Client**: VLLM客户端，与阿里云Qwen3-Omni-flash模型通信
- **Frame**: 视频帧，视频流中的单个图像
- **Service_Type**: 服务类型，包括社交语言转录文字、交通场景安全提醒

## Requirements

### Requirement 1: 视频流逐帧处理

**User Story:** 作为听障用户，我希望系统能够实时处理我眼镜摄像头的视频流，以便系统能够持续感知我周围的环境。

#### Acceptance Criteria

1. WHEN a video file is provided, THE Video_Stream_Processor SHALL treat it as a stream and process frames sequentially
2. WHEN processing video stream, THE Video_Stream_Processor SHALL extract each frame for analysis
3. WHEN a frame is extracted, THE Video_Stream_Processor SHALL pass it to the Scene_Analyzer within 100ms
4. IF frame extraction fails, THEN THE Video_Stream_Processor SHALL skip the corrupted frame and continue processing

### Requirement 2: VLLM模型调用与场景分析

**User Story:** 作为听障用户，我希望系统能够智能分析我所处的环境，以便系统能够理解我的处境并提供相应帮助。

#### Acceptance Criteria

1. WHEN a frame is received, THE Scene_Analyzer SHALL call the VLLM API with stream mode enabled
2. THE VLLM_Client SHALL use the Qwen3-Omni-flash model via DashScope compatible API
3. THE VLLM_Client SHALL set the stream parameter to True for all API calls
4. WHEN calling VLLM API, THE Scene_Analyzer SHALL request analysis of: scene description, objects present, user location, and user state
5. WHEN VLLM response is received, THE Scene_Analyzer SHALL parse the streaming response in real-time
6. IF VLLM API call fails, THEN THE Scene_Analyzer SHALL retry up to 3 times with exponential backoff

### Requirement 3: 场景信息输出

**User Story:** 作为听障用户，我希望系统能够识别我所处的场景类型和状态，以便系统能够提供针对性的服务。

#### Acceptance Criteria

1. WHEN analyzing a frame, THE Scene_Analyzer SHALL output a brief scene description
2. WHEN analyzing a frame, THE Scene_Analyzer SHALL identify and list objects present in the scene
3. WHEN analyzing a frame, THE Scene_Analyzer SHALL determine user location (e.g., meeting room, bar, street, restaurant)
4. WHEN analyzing a frame, THE Scene_Analyzer SHALL determine user state (e.g., walking, cycling, driving)
5. THE Scene_Analyzer SHALL format all outputs in a structured JSON format

### Requirement 4: 服务需求判断

**User Story:** 作为听障用户，我希望系统能够智能判断我是否需要帮助，以便在需要时主动为我提供服务。

#### Acceptance Criteria

1. WHEN scene analysis is complete, THE Service_Decision_Engine SHALL determine if service is needed
2. THE Service_Decision_Engine SHALL output only "是" or "否" for service need determination
3. WHEN service is needed, THE Service_Decision_Engine SHALL identify the required service type
4. THE Service_Decision_Engine SHALL support service types: social language transcription (社交语言转录文字) and traffic safety alerts (交通场景安全提醒)
5. WHEN user is in social interaction scenarios, THE Service_Decision_Engine SHALL recommend social language transcription service
6. WHEN user is in traffic scenarios (walking, cycling, driving on streets), THE Service_Decision_Engine SHALL recommend traffic safety alert service

### Requirement 5: 能耗与记忆管理

**User Story:** 作为听障用户，我希望系统能够长时间稳定运行而不会过度消耗电量或内存，以便我可以全天候使用这个服务。

#### Acceptance Criteria

1. THE Memory_Manager SHALL implement a sliding window mechanism for context storage
2. THE Memory_Manager SHALL limit stored context to the most recent N frames (configurable)
3. WHEN memory usage exceeds threshold, THE Memory_Manager SHALL automatically purge oldest context data
4. THE Video_Stream_Processor SHALL support configurable frame sampling rate to reduce processing load
5. WHEN system detects low activity scenes, THE Video_Stream_Processor SHALL reduce analysis frequency to conserve energy
6. THE Memory_Manager SHALL serialize and deserialize context data efficiently using JSON format

### Requirement 6: 极速响应

**User Story:** 作为听障用户，我希望系统能够快速响应环境变化，以便我能够及时获得必要的帮助。

#### Acceptance Criteria

1. THE Video_Stream_Processor SHALL process frames asynchronously to maximize throughput
2. THE Scene_Analyzer SHALL utilize streaming API responses to minimize latency
3. WHEN processing a frame, THE system SHALL complete service decision within 500ms end-to-end
4. THE system SHALL support concurrent processing of multiple frames when hardware permits
