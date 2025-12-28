import os
import time
import subprocess
import asyncio
import threading
import json
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
import dashscope
from dashscope.audio.asr import *
from datetime import datetime

from models import TranscriptSegment, EmotionResult
from tom_analyzer import SimToMAnalyzer, ToMConfig
from text_corrector import TextCorrector, CorrectorConfig

# API配置
dashscope.api_key = "sk-471ef1d9f6734e79a4186adec3660bdd"
dashscope.base_websocket_api_url = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'


def get_timestamp():
    now = datetime.now()
    return now.strftime("[%Y-%m-%d %H:%M:%S.%f]")


def extract_audio_from_video(video_path: str, output_wav_path: str) -> bool:
    """使用 ffmpeg 从视频中提取音频并转换为 WAV 格式"""
    print(f"{get_timestamp()} 正在从视频提取音频...")
    
    ffmpeg_path = r'D:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe'
    
    cmd = [
        ffmpeg_path, '-y',
        '-i', video_path,
        '-vn',
        '-ar', '16000',
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        output_wav_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg 错误: {result.stderr}")
            return False
        print(f"{get_timestamp()} 音频提取完成: {output_wav_path}")
        return True
    except FileNotFoundError:
        print("错误: 未找到 ffmpeg")
        return False


@dataclass
class ProcessedSentence:
    """处理后的句子"""
    original_text: str           # 原始识别文本
    corrected_text: str          # 修正后文本
    corrections: list            # 修正列表
    emotion_result: Optional[EmotionResult] = None  # ToM分析结果
    start_time: float = 0.0
    end_time: float = 0.0
    speaker_id: Optional[str] = None  # 说话人ID (person1, person2, ...)
    bbox: Optional[List[float]] = None  # 说话人边界框


@dataclass
class SpeakerInfo:
    """说话人信息（来自远程检测）"""
    step: int
    timestamp: float
    speaker_id: Optional[str]
    persons: List[Dict[str, Any]] = field(default_factory=list)


class SpeakerTracker:
    """说话人追踪器 - 根据时间戳匹配说话人"""
    
    def __init__(self, step_duration: float = 0.5):
        self.step_duration = step_duration
        self.speaker_info_list: List[SpeakerInfo] = []
        self.output_dir: Optional[str] = None
    
    def load_from_results(self, results: List[Dict]):
        """从远程检测结果加载说话人信息"""
        self.speaker_info_list = []
        for r in results:
            info = SpeakerInfo(
                step=r.get("step", 0),
                timestamp=r.get("timestamp", 0.0),
                speaker_id=r.get("speaker_id"),
                persons=r.get("persons", [])
            )
            self.speaker_info_list.append(info)
        
        print(f"{get_timestamp()} [SpeakerTracker] 加载了 {len(self.speaker_info_list)} 个step的说话人信息")
    
    def load_from_directory(self, output_dir: str):
        """从输出目录加载说话人信息"""
        self.output_dir = output_dir
        self.speaker_info_list = []
        
        # 查找所有 output_step_*.json 文件
        if not os.path.exists(output_dir):
            print(f"{get_timestamp()} [SpeakerTracker] 目录不存在: {output_dir}")
            return
        
        step = 1
        while True:
            filepath = os.path.join(output_dir, f"output_step_{step}.json")
            if not os.path.exists(filepath):
                break
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                info = SpeakerInfo(
                    step=data.get("step", step),
                    timestamp=data.get("timestamp", step * self.step_duration),
                    speaker_id=data.get("speaker_id"),
                    persons=data.get("persons", [])
                )
                self.speaker_info_list.append(info)
            except Exception as e:
                print(f"{get_timestamp()} [SpeakerTracker] 加载失败 {filepath}: {e}")
            
            step += 1
        
        print(f"{get_timestamp()} [SpeakerTracker] 从目录加载了 {len(self.speaker_info_list)} 个step的说话人信息")
    
    def get_speaker_at_time(self, timestamp: float) -> Optional[SpeakerInfo]:
        """根据时间戳获取说话人信息"""
        if not self.speaker_info_list:
            return None
        
        # 找到对应的step
        step = int(timestamp / self.step_duration) + 1
        
        for info in self.speaker_info_list:
            if info.step == step:
                return info
        
        # 如果没有精确匹配，找最近的
        closest = None
        min_diff = float('inf')
        for info in self.speaker_info_list:
            diff = abs(info.timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest = info
        
        return closest
    
    def get_speaker_id_at_time(self, timestamp: float) -> Optional[str]:
        """根据时间戳获取说话人ID"""
        info = self.get_speaker_at_time(timestamp)
        return info.speaker_id if info else None
    
    def get_person_bbox_at_time(self, timestamp: float, person_id: str) -> Optional[List[float]]:
        """根据时间戳和person_id获取边界框"""
        info = self.get_speaker_at_time(timestamp)
        if not info:
            return None
        
        for person in info.persons:
            if person.get("label") == person_id:
                return person.get("bbox")
        
        return None


class AsyncProcessor:
    """
    异步处理器
    在后台线程中运行，处理:
    1. 文本修正 - 结合上下文修正识别错误
    2. ToM分析 - 情感和意图分析
    3. 说话人追踪 - 根据时间戳匹配说话人
    """
    
    def __init__(self, api_key: str, speaker_tracker: Optional[SpeakerTracker] = None):
        # 文本修正器
        self.corrector = TextCorrector(CorrectorConfig(api_key=api_key))
        # ToM分析器
        self.tom_analyzer = SimToMAnalyzer(ToMConfig(api_key=api_key))
        # 说话人追踪器
        self.speaker_tracker = speaker_tracker
        
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: threading.Thread = None
        self._context_history: list[str] = []  # 上下文历史
        self._max_context = 5
        
        # 结果回调
        self.on_correction: Optional[Callable[[ProcessedSentence], None]] = None
        self.on_tom_result: Optional[Callable[[ProcessedSentence], None]] = None
        
    def start(self):
        """启动后台事件循环"""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        
    def process_sentence(self, text: str, start_time: float, end_time: float):
        """
        异步处理句子（不阻塞）
        1. 先进行文本修正
        2. 修正完成后进行ToM分析
        """
        if self._loop is None or not text.strip():
            return
            
        asyncio.run_coroutine_threadsafe(
            self._do_process(text, start_time, end_time),
            self._loop
        )
        
    async def _do_process(self, text: str, start_time: float, end_time: float):
        """执行异步处理"""
        # 获取说话人信息
        speaker_id = None
        bbox = None
        if self.speaker_tracker:
            mid_time = (start_time + end_time) / 2
            speaker_id = self.speaker_tracker.get_speaker_id_at_time(mid_time)
            if speaker_id and speaker_id != "OUT_OF_FRAME":
                bbox = self.speaker_tracker.get_person_bbox_at_time(mid_time, speaker_id)
        
        sentence = ProcessedSentence(
            original_text=text,
            corrected_text=text,
            corrections=[],
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_id,
            bbox=bbox
        )
        
        # 获取上下文
        context = self._context_history[-self._max_context:]
        context_str = " | ".join(context)
        
        try:
            # Step 1: 文本修正
            correction_result = await self.corrector.correct(text, context)
            sentence.corrected_text = correction_result["corrected"]
            sentence.corrections = correction_result.get("corrections", [])
            
            # 回调通知修正完成
            if self.on_correction:
                self.on_correction(sentence)
            
            # 更新上下文（使用修正后的文本）
            self._context_history.append(sentence.corrected_text)
            if len(self._context_history) > self._max_context:
                self._context_history.pop(0)
            
            # Step 2: ToM分析（使用修正后的文本）
            segment = TranscriptSegment(
                text=sentence.corrected_text,
                start_time=start_time,
                end_time=end_time
            )
            sentence.emotion_result = await self.tom_analyzer.analyze(segment, context_str)
            
            # 回调通知ToM分析完成
            if self.on_tom_result:
                self.on_tom_result(sentence)
                
        except Exception as e:
            print(f"{get_timestamp()} [处理错误] {e}")
            
    def stop(self):
        """停止后台事件循环"""
        if self._loop:
            # 关闭资源
            if self.corrector:
                asyncio.run_coroutine_threadsafe(self.corrector.close(), self._loop)
            if self.tom_analyzer:
                asyncio.run_coroutine_threadsafe(self.tom_analyzer.close(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)


# 全局处理器
processor: AsyncProcessor = None


def on_text_corrected(sentence: ProcessedSentence):
    """文本修正完成回调"""
    speaker_str = f"[{sentence.speaker_id}]" if sentence.speaker_id else "[未知]"
    
    if sentence.corrections:
        print(f"\n{get_timestamp()} [文本修正] {speaker_str}")
        print(f"  原文: {sentence.original_text}")
        print(f"  修正: {sentence.corrected_text}")
        for c in sentence.corrections:
            print(f"    - '{c.get('original')}' → '{c.get('corrected')}' ({c.get('reason')})")
    else:
        print(f"\n{get_timestamp()} [转录] {speaker_str} {sentence.corrected_text}")


def on_tom_completed(sentence: ProcessedSentence):
    """ToM分析完成回调"""
    result = sentence.emotion_result
    speaker_str = f"[{sentence.speaker_id}]" if sentence.speaker_id else "[未知]"
    
    if result:
        print(f"\n{get_timestamp()} [ToM分析] {speaker_str}")
        print(f"  文本: {sentence.corrected_text}")
        print(f"  情感: {result.emotion.value} (置信度: {result.confidence:.2f})")
        print(f"  意图: {result.speaker_intent}")
        print(f"  心理: {result.mental_state}")
        if result.suggested_response:
            print(f"  建议: {result.suggested_response}")
        if sentence.bbox:
            print(f"  位置: {sentence.bbox}")


class Callback(RecognitionCallback):
    """语音识别回调"""
    
    def __init__(self):
        self._sentence_start_time = time.time()
        
    def on_complete(self) -> None:
        print(f"\n{get_timestamp()} ===== 识别完成 =====")

    def on_error(self, result: RecognitionResult) -> None:
        print('Recognition task_id: ', result.request_id)
        print('Recognition error: ', result.message)

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if 'text' in sentence:
            text = sentence['text']
            print(f"{get_timestamp()} [实时] {text}")
            
            # 句子结束时，异步处理
            if RecognitionResult.is_sentence_end(sentence):
                end_time = time.time()
                print(f"{get_timestamp()} [句子结束] {text}")
                
                # 异步处理：文本修正 + ToM分析
                if processor and text.strip():
                    processor.process_sentence(
                        text, 
                        self._sentence_start_time, 
                        end_time
                    )
                
                self._sentence_start_time = time.time()


def transcribe_video(video_path: str):
    """从视频文件进行语音识别 + 文本修正 + ToM分析"""
    global processor
    
    wav_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
    
    # 步骤1: 提取音频
    if not extract_audio_from_video(video_path, wav_path):
        raise Exception("音频提取失败")
    
    # 步骤2: 启动异步处理器
    processor = AsyncProcessor(api_key=dashscope.api_key)
    processor.on_correction = on_text_corrected
    processor.on_tom_result = on_tom_completed
    processor.start()
    print(f"{get_timestamp()} 异步处理器已启动（文本修正 + ToM分析）")
    
    # 步骤3: 语音识别
    callback = Callback()
    recognition = Recognition(
        model='qwen3-asr-flash-realtime',
        format='wav',
        sample_rate=16000,
        callback=callback
    )
    
    recognition.start()
    
    try:
        file_size = os.path.getsize(wav_path)
        if file_size == 0:
            raise Exception('音频文件为空')
        
        print(f"{get_timestamp()} 开始发送音频数据，文件大小: {file_size} bytes")
        print("=" * 60)
        
        with open(wav_path, 'rb') as f:
            while True:
                audio_data = f.read(3200)
                if not audio_data:
                    break
                recognition.send_audio_frame(audio_data)
                time.sleep(0.1)
                
    except Exception as e:
        recognition.stop()
        processor.stop()
        raise e
    
    recognition.stop()
    
    # 等待异步处理完成
    print(f"\n{get_timestamp()} 等待异步处理完成...")
    time.sleep(5)
    processor.stop()
    
    print("=" * 60)
    print(
        '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
        .format(
            recognition.get_last_request_id(),
            recognition.get_first_package_delay(),
            recognition.get_last_package_delay(),
        ))


if __name__ == '__main__':
    video_file = r"E:\vid_data\VID_20251225_191155.mp4"
    transcribe_video(video_file)


# ==================== 带说话人信息的转录 ====================
def transcribe_with_speaker_info(speaker_results: List[Dict], output_dir: str, 
                                  audio_path: Optional[str] = None):
    """
    带说话人信息的转录
    
    Args:
        speaker_results: 远程检测返回的结果列表
            [{"step": 1, "timestamp": 0.5, "speaker_id": "person1", "persons": [...]}]
        output_dir: 输出目录（包含output_step_*.json文件）
        audio_path: 音频文件路径（可选，如果不提供则从output_dir读取）
    """
    global processor
    
    print(f"{get_timestamp()} ===== 开始带说话人信息的转录 =====")
    print(f"  输出目录: {output_dir}")
    print(f"  结果数量: {len(speaker_results)}")
    
    # 创建说话人追踪器
    speaker_tracker = SpeakerTracker()
    
    if speaker_results:
        speaker_tracker.load_from_results(speaker_results)
    else:
        speaker_tracker.load_from_directory(output_dir)
    
    # 查找音频文件
    if audio_path is None:
        # 尝试从output_dir查找
        possible_paths = [
            os.path.join(output_dir, "audio.wav"),
            os.path.join(output_dir, "audio", "audio.wav"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                audio_path = p
                break
    
    if audio_path is None or not os.path.exists(audio_path):
        print(f"{get_timestamp()} [错误] 未找到音频文件")
        return
    
    print(f"  音频文件: {audio_path}")
    
    # 启动异步处理器（带说话人追踪）
    processor = AsyncProcessor(api_key=dashscope.api_key, speaker_tracker=speaker_tracker)
    processor.on_correction = on_text_corrected
    processor.on_tom_result = on_tom_completed
    processor.start()
    print(f"{get_timestamp()} 异步处理器已启动（文本修正 + ToM分析 + 说话人追踪）")
    
    # 语音识别
    callback = Callback()
    recognition = Recognition(
        model='fun-asr-realtime',
        format='wav',
        sample_rate=16000,
        callback=callback
    )
    
    recognition.start()
    
    try:
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise Exception('音频文件为空')
        
        print(f"{get_timestamp()} 开始发送音频数据，文件大小: {file_size} bytes")
        print("=" * 60)
        
        with open(audio_path, 'rb') as f:
            while True:
                audio_data = f.read(3200)
                if not audio_data:
                    break
                recognition.send_audio_frame(audio_data)
                time.sleep(0.1)
                
    except Exception as e:
        recognition.stop()
        processor.stop()
        raise e
    
    recognition.stop()
    
    # 等待异步处理完成
    print(f"\n{get_timestamp()} 等待异步处理完成...")
    time.sleep(5)
    processor.stop()
    
    print("=" * 60)
    print(f"{get_timestamp()} ===== 转录完成 =====")


def transcribe_realtime_with_speaker(speaker_tracker: SpeakerTracker):
    """
    实时转录（带说话人追踪）
    用于流式处理场景
    
    Args:
        speaker_tracker: 说话人追踪器（会实时更新）
    """
    global processor
    
    # 启动异步处理器
    processor = AsyncProcessor(api_key=dashscope.api_key, speaker_tracker=speaker_tracker)
    processor.on_correction = on_text_corrected
    processor.on_tom_result = on_tom_completed
    processor.start()
    
    return processor
