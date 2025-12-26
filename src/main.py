"""
听障辅助Agent命令行入口
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .models import AgentConfig, ProcessorConfig, VLLMConfig, MemoryConfig
from .agent import HearingImpairedAgent


def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def load_config(config_path: str) -> AgentConfig:
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return AgentConfig.from_dict(data)


def create_default_config(api_key: str) -> AgentConfig:
    """创建默认配置"""
    return AgentConfig(
        processor_config=ProcessorConfig(
            sample_rate=1,
            max_fps=30.0,
        ),
        vllm_config=VLLMConfig(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            model="qwen-omni-turbo",
            max_retries=3,
            timeout=30.0,
        ),
        memory_config=MemoryConfig(
            max_context_frames=100,
            memory_threshold_mb=100.0,
        ),
        adaptive_sampling=True,
    )


async def run_agent(video_path: str, config: AgentConfig, output_file: str = None) -> None:
    """运行Agent处理视频"""
    agent = HearingImpairedAgent(config)
    
    results = []
    
    try:
        print(f"开始处理视频: {video_path}", file=sys.stderr)
        
        async for output in agent.run(video_path):
            # 输出到控制台
            result = output.to_dict()
            
            # 打印关键信息
            print(f"\n=== 帧 {output.frame_id} (时间: {output.timestamp:.2f}s) ===")
            print(f"场景描述: {output.scene_analysis.scene_description}")
            print(f"检测物体: {', '.join(output.scene_analysis.objects)}")
            print(f"用户位置: {output.scene_analysis.user_location.value}")
            print(f"用户状态: {output.scene_analysis.user_state.value}")
            print(f"是否需要服务: {output.service_decision.needs_service_str()}")
            
            if output.service_decision.needs_service:
                service_name = {
                    "social_transcription": "社交语言转录文字",
                    "traffic_safety_alert": "交通场景安全提醒",
                }.get(output.service_decision.service_type.value, "未知服务")
                print(f"服务类型: {service_name}")
                print(f"优先级: {output.service_decision.priority}")
            
            results.append(result)
        
        # 保存结果到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_file}", file=sys.stderr)
        
        # 打印统计信息
        stats = agent.get_statistics()
        print(f"\n=== 处理完成 ===", file=sys.stderr)
        print(f"处理帧数: {stats['processed_frames']}", file=sys.stderr)
        print(f"内存使用: {stats['memory_usage_mb']:.2f} MB", file=sys.stderr)
        
    finally:
        await agent.close()


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='听障辅助Agent - AI眼镜视频分析与主动服务决策',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.main video.mp4 --api-key YOUR_API_KEY
  python -m src.main video.mp4 --config config.json
  python -m src.main video.mp4 --api-key YOUR_API_KEY --output results.json
        """
    )
    
    parser.add_argument(
        'video',
        help='视频文件路径'
    )
    
    parser.add_argument(
        '--api-key',
        help='DashScope API密钥',
        default=''
    )
    
    parser.add_argument(
        '--config',
        help='配置文件路径 (JSON格式)',
        default=None
    )
    
    parser.add_argument(
        '--output', '-o',
        help='输出结果文件路径 (JSON格式)',
        default=None
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=1,
        help='帧采样率，每N帧处理一次 (默认: 1)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='最大存储帧数 (默认: 100)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    
    return parser.parse_args()


def main() -> None:
    """主入口函数"""
    args = parse_args()
    
    # 配置日志
    setup_logging(args.verbose)
    
    # 检查视频文件
    if not Path(args.video).exists():
        print(f"错误: 视频文件不存在: {args.video}", file=sys.stderr)
        sys.exit(1)
    
    # 加载或创建配置
    if args.config:
        config = load_config(args.config)
    else:
        if not args.api_key:
            print("错误: 请提供 --api-key 或 --config 参数", file=sys.stderr)
            sys.exit(1)
        config = create_default_config(args.api_key)
    
    # 应用命令行参数覆盖
    config.processor_config.sample_rate = args.sample_rate
    config.memory_config.max_context_frames = args.max_frames
    
    # 运行Agent
    try:
        asyncio.run(run_agent(args.video, config, args.output))
    except KeyboardInterrupt:
        print("\n处理被用户中断", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
