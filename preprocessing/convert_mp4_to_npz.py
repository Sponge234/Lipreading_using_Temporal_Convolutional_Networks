#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MP4视频转换为NPZ格式脚本
用于将原始MP4视频转换为模型可识别的NPZ格式
支持两种模式：
1. 使用预计算的人脸关键点（推荐）
2. 自动检测人脸关键点（需要dlib）
"""

import os
import sys
import cv2
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(__file__))
from dataloader import AVSRDataLoader
from utils import save2npz


def detect_landmarks_dlib(frame, detector, predictor):
    """使用dlib检测人脸关键点
    
    Args:
        frame: BGR图像
        detector: dlib人脸检测器
        predictor: dlib关键点预测器
    
    Returns:
        landmarks: 68个关键点坐标，shape=(68, 2)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    
    if len(dets) == 0:
        return None
    
    # 取最大的人脸
    shape = predictor(gray, dets[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks


def detect_landmarks_mediapipe(frame, face_mesh):
    """使用MediaPipe检测人脸关键点
    
    Args:
        frame: BGR图像
        face_mesh: MediaPipe人脸网格模型
    
    Returns:
        landmarks: 68个关键点坐标，shape=(68, 2)
    """
    import mediapipe as mp
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
    
    # MediaPipe有468个关键点，需要映射到68个关键点
    # 这里简化处理，提取嘴部区域的关键点
    h, w = frame.shape[:2]
    mp_landmarks = results.multi_face_landmarks[0]
    
    # 映射MediaPipe到68点格式（简化版）
    # MediaPipe嘴部关键点索引: 61-308
    # 这里创建一个简化的68点格式
    landmarks = np.zeros((68, 2))
    
    # 使用MediaPipe的关键点映射
    # 下巴轮廓 (0-16)
    chin_indices = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67]
    for i, idx in enumerate(chin_indices[:17]):
        landmarks[i] = [mp_landmarks.landmark[idx].x * w, mp_landmarks.landmark[idx].y * h]
    
    # 眉毛 (17-27)
    left_eyebrow = [70, 63, 105, 66, 107]
    right_eyebrow = [336, 296, 334, 293, 300]
    for i, idx in enumerate(left_eyebrow):
        landmarks[17+i] = [mp_landmarks.landmark[idx].x * w, mp_landmarks.landmark[idx].y * h]
    for i, idx in enumerate(right_eyebrow):
        landmarks[22+i] = [mp_landmarks.landmark[idx].x * w, mp_landmarks.landmark[idx].y * h]
    
    # 鼻子 (27-35)
    nose_indices = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
    for i, idx in enumerate(nose_indices[:10]):
        landmarks[27+i] = [mp_landmarks.landmark[idx].x * w, mp_landmarks.landmark[idx].y * h]
    
    # 眼睛 (36-47)
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    for i, idx in enumerate(left_eye[:6]):
        landmarks[36+i] = [mp_landmarks.landmark[idx].x * w, mp_landmarks.landmark[idx].y * h]
    for i, idx in enumerate(right_eye[:6]):
        landmarks[42+i] = [mp_landmarks.landmark[idx].x * w, mp_landmarks.landmark[idx].y * h]
    
    # 嘴巴 (48-67)
    mouth_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 324, 318]
    mouth_inner = [78, 95, 88, 86, 80, 77, 13, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 268]
    for i, idx in enumerate(mouth_outer[:12]):
        landmarks[48+i] = [mp_landmarks.landmark[idx].x * w, mp_landmarks.landmark[idx].y * h]
    for i, idx in enumerate(mouth_inner[:8]):
        landmarks[60+i] = [mp_landmarks.landmark[idx].x * w, mp_landmarks.landmark[idx].y * h]
    
    return landmarks


def process_video_with_landmarks(video_path, landmarks_path, dataloader, output_path):
    """使用预计算的关键点处理视频
    
    Args:
        video_path: 视频文件路径
        landmarks_path: 关键点文件路径
        dataloader: AVSRDataLoader实例
        output_path: 输出NPZ文件路径
    
    Returns:
        success: 是否成功
    """
    try:
        sequence = dataloader.load_data("video", video_path, landmarks_path)
        if sequence is not None:
            save2npz(output_path, data=sequence)
            return True
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    return False


def process_video_auto_detect(video_path, dataloader, output_path, detector_type='mediapipe'):
    """自动检测人脸关键点并处理视频
    
    Args:
        video_path: 视频文件路径
        dataloader: AVSRDataLoader实例
        output_path: 输出NPZ文件路径
        detector_type: 检测器类型 ('dlib' 或 'mediapipe')
    
    Returns:
        success: 是否成功
    """
    # 初始化检测器
    if detector_type == 'dlib':
        try:
            import dlib
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        except ImportError:
            print("Error: dlib not installed. Install with: pip install dlib")
            return False
    elif detector_type == 'mediapipe':
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        except ImportError:
            print("Error: mediapipe not installed. Install with: pip install mediapipe")
            return False
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # 检测每一帧的关键点
    landmarks_list = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm(range(frame_count), desc="Detecting landmarks"):
        ret, frame = cap.read()
        if not ret:
            break
        
        if detector_type == 'dlib':
            landmarks = detect_landmarks_dlib(frame, detector, predictor)
        else:
            landmarks = detect_landmarks_mediapipe(frame, face_mesh)
        
        landmarks_list.append(landmarks)
    
    cap.release()
    
    # 检查是否所有帧都检测到关键点
    valid_count = sum(1 for lm in landmarks_list if lm is not None)
    if valid_count < len(landmarks_list) * 0.5:
        print(f"Warning: Only {valid_count}/{len(landmarks_list)} frames have valid landmarks")
        return False
    
    # 使用检测到的关键点处理视频
    try:
        sequence = dataloader.preprocess(video_path, landmarks_list)
        if sequence is not None:
            save2npz(output_path, data=sequence)
            return True
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Convert MP4 videos to NPZ format for lipreading')
    
    # 输入输出路径
    parser.add_argument('--video-direc', required=True, help='Directory containing MP4 videos')
    parser.add_argument('--landmark-direc', default=None, help='Directory containing landmark pkl files (optional)')
    parser.add_argument('--output-direc', required=True, help='Output directory for NPZ files')
    
    # 处理模式
    parser.add_argument('--auto-detect', action='store_true', 
                       help='Auto-detect face landmarks (requires dlib or mediapipe)')
    parser.add_argument('--detector', default='mediapipe', choices=['dlib', 'mediapipe'],
                       help='Landmark detector type (default: mediapipe)')
    
    # 其他参数
    parser.add_argument('--convert-gray', default=True, action='store_true',
                       help='Convert to grayscale (default: True)')
    parser.add_argument('--filename-list', default=None,
                       help='File containing list of video filenames to process (optional)')
    parser.add_argument('--video-extension', default='.mp4', help='Video file extension')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_direc, exist_ok=True)
    
    # 初始化dataloader
    dataloader = AVSRDataLoader(convert_gray=args.convert_gray)
    
    # 获取视频文件列表
    if args.filename_list:
        with open(args.filename_list, 'r') as f:
            video_files = [line.strip() for line in f.readlines()]
    else:
        video_files = [f for f in os.listdir(args.video_direc) 
                      if f.endswith(args.video_extension)]
    
    print(f"Found {len(video_files)} videos to process")
    
    # 处理每个视频
    success_count = 0
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(args.video_direc, video_file)
        
        # 构造输出文件名
        output_name = os.path.splitext(video_file)[0] + '.npz'
        output_path = os.path.join(args.output_direc, output_name)
        
        # 跳过已存在的文件
        if os.path.exists(output_path):
            print(f"Skipping {video_file} (already exists)")
            success_count += 1
            continue
        
        # 处理视频
        if args.auto_detect:
            # 自动检测模式
            success = process_video_auto_detect(
                video_path, dataloader, output_path, detector_type=args.detector
            )
        else:
            # 使用预计算关键点模式
            if args.landmark_direc is None:
                print(f"Error: --landmark-direc required when not using --auto-detect")
                continue
            
            landmark_name = os.path.splitext(video_file)[0] + '.pkl'
            landmark_path = os.path.join(args.landmark_direc, landmark_name)
            
            if not os.path.exists(landmark_path):
                print(f"Warning: Landmark file not found for {video_file}")
                continue
            
            success = process_video_with_landmarks(
                video_path, landmark_path, dataloader, output_path
            )
        
        if success:
            success_count += 1
    
    print(f"\nProcessing complete: {success_count}/{len(video_files)} videos converted successfully")


if __name__ == '__main__':
    main()
