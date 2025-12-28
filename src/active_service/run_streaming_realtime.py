#!/usr/bin/env python3
"""
流式实时处理脚本 - 用于远程服务器

功能:
1. 接收流式图像和音频输入（而非视频文件）
2. 使用已有的YOLO检测结果
3. 运行EZ-VSL音源定位
4. 输出说话人归属结果

部署到服务器: /root/autodl-tmp/EZ-VSL/run_streaming_realtime.py

使用方法:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ezvsl && cd EZ-VSL && \
python run_streaming_realtime.py \
    --yolo-json ../streaming_sessions/xxx/persons.json \
    --frames-dir ../streaming_sessions/xxx/frames \
    --audio-path ../streaming_sessions/xxx/audio/audio.wav \
    --output-dir ../streaming_sessions/xxx/attribution \
    --hf-token Fitz123 \
    --api-key sk_176df79b36fb44c4b92969e798265dc9
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import cv2

# 添加模块路径
EZVSL_PATH = "/root/autodl-tmp/EZ-VSL"
sys.path.insert(0, EZVSL_PATH)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STEP_DURATION = 0.5  # 每个step的时长（秒）


def normalize_bbox(bbox: List, image_size: List[int] = None) -> List[float]:
    """
    归一化边界框坐标
    
    Args:
        bbox: [x1, y1, x2, y2] 像素坐标或已归一化坐标
        image_size: [width, height] 图像尺寸
    
    Returns:
        归一化后的 [x1, y1, x2, y2]
    """
    if not bbox or len(bbox) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    
    # 检查是否已经归一化（所有值都在0-1之间）
    if all(0 <= v <= 1 for v in bbox):
        return [float(v) for v in bbox]
    
    # 需要归一化
    if image_size and len(image_size) == 2:
        w, h = image_size
        return [
            bbox[0] / w,
            bbox[1] / h,
            bbox[2] / w,
            bbox[3] / h
        ]
    
    # 无法归一化，假设是1920x1080
    return [
        bbox[0] / 1920,
        bbox[1] / 1080,
        bbox[2] / 1920,
        bbox[3] / 1080
    ]


def load_yolo_json(json_path: str) -> Dict:
    """加载并归一化YOLO检测JSON"""
    logger.info(f"Loading YOLO JSON: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图像尺寸
    video_info = data.get("video_info", {})
    image_size = [
        video_info.get("width", 1920),
        video_info.get("height", 1080)
    ]
    
    # 归一化所有bbox
    frames = data.get("frames", [])
    for frame in frames:
        for person in frame.get("persons", []):
            bbox = person.get("bbox", [])
            person["bbox"] = normalize_bbox(bbox, image_size)
    
    logger.info(f"Loaded {len(frames)} frames, image_size={image_size}")
    return data


def split_yolo_to_steps(yolo_data: Dict, output_dir: str) -> int:
    """将YOLO数据拆分为单独的step文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    frames = yolo_data.get("frames", [])
    
    for frame in frames:
        step = frame.get("step", 0)
        if step == 0:
            continue
        
        output_data = {
            "step": step,
            "frame": frame.get("frame", 0),
            "persons": frame.get("persons", []),
            "count": frame.get("count", 0)
        }
        
        filepath = os.path.join(output_dir, f"step_{step:03d}_yolo.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    
    logger.info(f"Split {len(frames)} YOLO frames to {output_dir}")
    return len(frames)


def run_diart_diarization(audio_path: str, output_dir: str,
                          hf_token: str, api_key: str) -> Dict[int, List[str]]:
    """运行说话人分割"""
    logger.info(f"Running diarization on {audio_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 尝试导入run_streaming_simulation中的函数
        from run_streaming_simulation import run_diart_diarization as _run_diart
        return _run_diart(audio_path, output_dir, hf_token, api_key)
    except ImportError:
        logger.warning("Cannot import run_streaming_simulation, using mock diarization")
        return mock_diarization(audio_path, output_dir)


def mock_diarization(audio_path: str, output_dir: str) -> Dict[int, List[str]]:
    """模拟说话人分割"""
    import wave
    
    try:
        with wave.open(audio_path, 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)
            audio_data = wav.readframes(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
    except Exception as e:
        logger.warning(f"Failed to read audio: {e}")
        duration = 30.0
        audio_array = None
    
    total_steps = int(duration / STEP_DURATION) + 1
    samples_per_step = int(16000 * STEP_DURATION)
    
    speakers_by_step = {}
    
    for step in range(1, total_steps + 1):
        if audio_array is not None:
            start_sample = (step - 1) * samples_per_step
            end_sample = min(start_sample + samples_per_step, len(audio_array))
            
            if start_sample < len(audio_array):
                segment = audio_array[start_sample:end_sample]
                energy = np.sqrt(np.mean(segment.astype(np.float32) ** 2))
                
                if energy > 500:
                    speakers_by_step[step] = ["SPEAKER_00"]
                else:
                    speakers_by_step[step] = []
            else:
                speakers_by_step[step] = []
        else:
            speakers_by_step[step] = ["SPEAKER_00"]
        
        # 保存JSON
        output_data = {
            "step": step,
            "active_speakers": speakers_by_step[step]
        }
        
        filepath = os.path.join(output_dir, f"step_{step:03d}_diart.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    
    return speakers_by_step


def generate_heatmaps(frames_dir: str, audio_path: str, heatmap_dir: str,
                      checkpoint_path: str, total_steps: int) -> bool:
    """生成EZ-VSL热力图"""
    logger.info(f"Generating heatmaps for {total_steps} steps")
    
    os.makedirs(heatmap_dir, exist_ok=True)
    
    try:
        from diart_multi_source.ezvsl_inference import EZVSLInference
        inferencer = EZVSLInference(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to load EZ-VSL: {e}")
        return False
    
    for step in range(1, total_steps + 1):
        frame_path = os.path.join(frames_dir, f"step_{step:03d}.jpg")
        
        if not os.path.exists(frame_path):
            logger.warning(f"Frame not found: {frame_path}")
            _save_default_heatmap(heatmap_dir, step)
            continue
        
        timestamp = (step - 1) * STEP_DURATION
        audio_start = max(0, timestamp - 1.5)
        
        try:
            result = inferencer.process_frame(
                frame_path, audio_path,
                audio_start=audio_start,
                audio_duration=3.0
            )
            
            heatmap_data = {
                "step": step,
                "timestamp": timestamp,
                "anchor": result["anchor"],
                "bbox": result["bbox"],
                "peak_energy": result["peak_energy"],
                "is_valid": result["is_valid"],
                "is_mock": False
            }
            
            filepath = os.path.join(heatmap_dir, f"step_{step:03d}_heatmap.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(heatmap_data, f, indent=2)
            
            logger.info(f"Step {step}: anchor=({result['anchor']['x']:.3f}, {result['anchor']['y']:.3f})")
            
        except Exception as e:
            logger.error(f"Step {step} failed: {e}")
            _save_default_heatmap(heatmap_dir, step)
    
    return True


def _save_default_heatmap(heatmap_dir: str, step: int):
    """保存默认热力图"""
    heatmap_data = {
        "step": step,
        "timestamp": (step - 1) * STEP_DURATION,
        "anchor": {"x": 0.5, "y": 0.5},
        "bbox": [0.3, 0.3, 0.7, 0.7],
        "peak_energy": 0.0,
        "is_valid": False,
        "is_mock": True
    }
    
    filepath = os.path.join(heatmap_dir, f"step_{step:03d}_heatmap.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(heatmap_data, f, indent=2)


def match_speakers(yolo_dir: str, heatmap_dir: str, diart_data: Dict[int, List[str]],
                   output_dir: str, total_steps: int) -> List[Dict]:
    """匹配说话人到人物"""
    logger.info("Matching speakers to persons")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # SPEAKER绑定状态
    speaker_to_person: Dict[str, Optional[str]] = {}
    assigned_persons: set = set()
    
    results = []
    
    for step in range(1, total_steps + 1):
        timestamp = step * STEP_DURATION
        
        # 读取YOLO数据
        yolo_file = os.path.join(yolo_dir, f"step_{step:03d}_yolo.json")
        yolo_data = {}
        if os.path.exists(yolo_file):
            with open(yolo_file, 'r') as f:
                yolo_data = json.load(f)
        
        persons = yolo_data.get("persons", [])
        
        # 读取热力图数据
        heatmap_file = os.path.join(heatmap_dir, f"step_{step:03d}_heatmap.json")
        heatmap_data = {}
        if os.path.exists(heatmap_file):
            with open(heatmap_file, 'r') as f:
                heatmap_data = json.load(f)
        
        anchor = heatmap_data.get("anchor", {"x": 0.5, "y": 0.5})
        
        # 获取diart说话人
        diart_speakers = diart_data.get(step, [])
        
        # 匹配逻辑
        speaker_id = None
        
        if diart_speakers:
            diart_speaker = diart_speakers[0]
            
            if diart_speaker in speaker_to_person:
                # 已绑定
                speaker_id = speaker_to_person[diart_speaker]
            else:
                # 未绑定，尝试匹配
                available_persons = [p for p in persons if p["label"] not in assigned_persons]
                
                for person in available_persons:
                    bbox = person.get("bbox", [0, 0, 1, 1])
                    if len(bbox) == 4:
                        x_min, y_min, x_max, y_max = bbox
                        if x_min <= anchor["x"] <= x_max and y_min <= anchor["y"] <= y_max:
                            speaker_id = person["label"]
                            speaker_to_person[diart_speaker] = speaker_id
                            assigned_persons.add(speaker_id)
                            break
                
                if speaker_id is None:
                    speaker_id = "OUT_OF_FRAME"
                    speaker_to_person[diart_speaker] = None
        
        # 构建输出
        speakers = []
        if speaker_id and speaker_id != "OUT_OF_FRAME":
            for p in persons:
                if p["label"] == speaker_id:
                    speakers.append({
                        "person_id": speaker_id,
                        "bbox": p["bbox"],
                        "confidence": p.get("confidence", 0.5)
                    })
                    break
        elif speaker_id == "OUT_OF_FRAME":
            speakers.append({
                "person_id": "OUT_OF_FRAME",
                "bbox": [-1, -1, -1, -1],
                "confidence": 0.0
            })
        
        output_data = {
            "step": step,
            "timestamp": timestamp,
            "speakers": speakers,
            "speaker_id": speaker_id
        }
        
        # 保存
        filepath = os.path.join(output_dir, f"output_step_{step}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        results.append(output_data)
        
        logger.info(f"Step {step}: speaker={speaker_id}")
    
    # 保存汇总
    summary = {
        "total_steps": total_steps,
        "results": results
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='流式实时处理')
    
    parser.add_argument('--yolo-json', type=str, required=True,
                       help='YOLO检测JSON文件路径')
    parser.add_argument('--frames-dir', type=str, required=True,
                       help='帧图像目录')
    parser.add_argument('--audio-path', type=str, required=True,
                       help='音频文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--hf-token', type=str, required=True,
                       help='HuggingFace token')
    parser.add_argument('--api-key', type=str, required=True,
                       help='pyannote API key')
    parser.add_argument('--checkpoint', type=str, 
                       default=os.path.join(EZVSL_PATH, "vggsound_10k/best.pth"),
                       help='EZ-VSL模型检查点路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    yolo_split_dir = os.path.join(args.output_dir, 'yolo_split')
    diart_dir = os.path.join(args.output_dir, 'diart')
    heatmap_dir = os.path.join(args.output_dir, 'heatmap')
    attribution_dir = os.path.join(args.output_dir, 'attribution')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Streaming Realtime Processing")
    logger.info("=" * 60)
    logger.info(f"YOLO JSON: {args.yolo_json}")
    logger.info(f"Frames: {args.frames_dir}")
    logger.info(f"Audio: {args.audio_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)
    
    # Step 1: 加载并归一化YOLO数据
    logger.info("\n[Step 1] Loading and normalizing YOLO data...")
    yolo_data = load_yolo_json(args.yolo_json)
    total_steps = split_yolo_to_steps(yolo_data, yolo_split_dir)
    
    # Step 2: 运行说话人分割
    logger.info("\n[Step 2] Running speaker diarization...")
    diart_data = run_diart_diarization(
        args.audio_path, diart_dir,
        args.hf_token, args.api_key
    )
    
    # Step 3: 生成热力图
    logger.info("\n[Step 3] Generating heatmaps...")
    if os.path.exists(args.checkpoint):
        generate_heatmaps(
            args.frames_dir, args.audio_path, heatmap_dir,
            args.checkpoint, total_steps
        )
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}, skipping heatmap generation")
        # 生成默认热力图
        for step in range(1, total_steps + 1):
            _save_default_heatmap(heatmap_dir, step)
    
    # Step 4: 匹配说话人
    logger.info("\n[Step 4] Matching speakers...")
    results = match_speakers(
        yolo_split_dir, heatmap_dir, diart_data,
        attribution_dir, total_steps
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Processing Complete!")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Output: {attribution_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
