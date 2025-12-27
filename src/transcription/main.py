"""
转录模块命令行入口
"""
import argparse
import asyncio
import logging
import sys

from .models import TranscriptionConfig, AudioConfig, SubtitleStyle
from .transcription_agent import TranscriptionAgent


def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='实时语音转录与情感分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.transcription.main video.mp4 --api-key YOUR_API_KEY
  python -m src.transcription.main video.mp4 --api-key YOUR_API_KEY --output output.mp4
  python -m src.transcription.main video.mp4 --api-key YOUR_API_KEY --no-emotion
        """
    )
    
    parser.add_argument('video', help='视频文件路径')
    parser.add_argument('--api-key', required=True, help='DashScope API密钥')
    parser.add_argument('--output', '-o', help='输出视频路径')
    parser.add_argument('--no-display', action='store_true', help='不显示窗口')
    parser.add_argument('--no-emotion', action='store_true', help='禁用情感分析')
    parser.add_argument('--no-intent', action='store_true', help='不显示说话者意图')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细日志')
    
    return parser.parse_args()


async def main_async() -> None:
    """异步主函数"""
    args = parse_args()
    setup_logging(args.verbose)
    
    # 创建配置
    config = TranscriptionConfig(
        api_key=args.api_key,
        enable_emotion=not args.no_emotion,
        show_emotion_indicator=not args.no_emotion,
        show_speaker_intent=not args.no_intent and not args.no_emotion,
    )
    
    # 创建Agent
    agent = TranscriptionAgent(config)
    
    print(f"开始处理视频: {args.video}")
    print("按 'q' 键退出")
    
    try:
        await agent.process_realtime(
            args.video,
            output_path=args.output,
            display=not args.no_display,
        )
        
        # 打印统计
        stats = agent.get_statistics()
        print(f"\n处理完成:")
        print(f"  转录片段数: {stats['total_segments']}")
        print(f"  总时长: {stats['total_duration']}")
        
    except KeyboardInterrupt:
        print("\n处理被中断")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def main() -> None:
    """主入口"""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
