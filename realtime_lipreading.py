#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时摄像头唇语检测系统
使用ResNet18-DCTCN模型进行实时唇语识别
"""

import os
import sys
import cv2
import json
import torch
import argparse
import numpy as np
from collections import deque
import time

# 添加项目路径
sys.path.append(os.path.dirname(__file__))
from lipreading.model import Lipreading
from lipreading.preprocess import *
from lipreading.dataloaders import get_preprocessing_pipelines


class RealTimeLipreading:
    """实时唇语检测类"""
    
    def __init__(self, config_path, model_path, label_path, 
                 buffer_size=29, use_mediapipe=True, device='auto'):
        """
        初始化实时唇语检测系统
        
        Args:
            config_path: 模型配置文件路径
            model_path: 预训练模型路径
            label_path: 标签文件路径
            buffer_size: 帧缓冲大小（默认29帧）
            use_mediapipe: 是否使用MediaPipe进行人脸检测
            device: 运行设备 ('auto', 'cuda', 'cpu')
        """
        # 设置设备
        self.device = self.setup_device(device)
        
        # 加载模型
        self.model = self.load_model(config_path, model_path)
        self.labels = self.load_labels(label_path)
        
        # 初始化人脸检测器
        self.use_mediapipe = use_mediapipe
        if use_mediapipe:
            self.init_mediapipe()
        else:
            self.init_opencv_detector()
        
        # 帧缓冲
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.mouth_buffer = deque(maxlen=buffer_size)
        
        # 预处理
        self.preprocessing = get_preprocessing_pipelines('video')['val']
        
        # 性能统计
        self.fps_counter = deque(maxlen=30)
        self.inference_times = deque(maxlen=10)
        
        # 预测结果
        self.current_prediction = ""
        self.confidence = 0.0
        self.top5_predictions = []
    
    def setup_device(self, device='auto'):
        """设置运行设备
        
        Args:
            device: 'auto', 'cuda', 或 'cpu'
        
        Returns:
            torch.device对象
        """
        if device == 'auto':
            # 自动检测
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                print("✓ 使用CPU (GPU不可用)")
        elif device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠ GPU不可用，切换到CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print("✓ 使用CPU")
        
        return device
        
    def load_model(self, config_path, model_path):
        """加载模型"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        densetcn_options = {
            'block_config': config['densetcn_block_config'],
            'growth_rate_set': config['densetcn_growth_rate_set'],
            'reduced_size': config['densetcn_reduced_size'],
            'kernel_size_set': config['densetcn_kernel_size_set'],
            'dilation_size_set': config['densetcn_dilation_size_set'],
            'squeeze_excitation': config['densetcn_se'],
            'dropout': config['densetcn_dropout'],
        }
        
        model = Lipreading(
            modality='video',
            num_classes=500,
            backbone_type=config['backbone_type'],
            relu_type=config['relu_type'],
            width_mult=config['width_mult'],
            densetcn_options=densetcn_options,
            use_boundary=config.get('use_boundary', False)
        )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"模型加载成功: {model_path}")
        
        # 移动模型到指定设备
        model = model.to(self.device)
        model.eval()
        
        # CPU优化：设置线程数
        if self.device.type == 'cpu':
            torch.set_num_threads(4)  # 使用4个CPU线程
            print("CPU优化: 使用4线程")
        
        return model
    
    def load_labels(self, label_path):
        """加载标签"""
        if not os.path.exists(label_path):
            return [f"word_{i}" for i in range(500)]
        
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    
    def init_mediapipe(self):
        """初始化MediaPipe人脸检测"""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe人脸检测器初始化成功")
        except ImportError:
            print("MediaPipe未安装，切换到OpenCV检测器")
            self.use_mediapipe = False
            self.init_opencv_detector()
    
    def init_opencv_detector(self):
        """初始化OpenCV人脸检测器"""
        # 使用Haar级联分类器
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 嘴部检测器
        mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
        if os.path.exists(mouth_cascade_path):
            self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
        else:
            self.mouth_cascade = None
        
        print("OpenCV人脸检测器初始化成功")
    
    def detect_face_mediapipe(self, frame):
        """使用MediaPipe检测人脸和嘴部区域"""
        import mediapipe as mp
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0]
        
        # 获取人脸边界框
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # 扩展边界框
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # 获取嘴部关键点（MediaPipe嘴部索引：61-308）
        mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 
                        291, 308, 415, 310, 311, 312, 324, 318,
                        78, 95, 88, 86, 80, 77, 13, 14, 317, 402]
        
        mouth_x = [landmarks.landmark[i].x * w for i in mouth_indices if i < len(landmarks.landmark)]
        mouth_y = [landmarks.landmark[i].y * h for i in mouth_indices if i < len(landmarks.landmark)]
        
        if len(mouth_x) > 0 and len(mouth_y) > 0:
            mouth_x_min, mouth_x_max = int(min(mouth_x)), int(max(mouth_x))
            mouth_y_min, mouth_y_max = int(min(mouth_y)), int(max(mouth_y))
            
            # 扩展嘴部区域
            mouth_padding_x = int((mouth_x_max - mouth_x_min) * 0.3)
            mouth_padding_y = int((mouth_y_max - mouth_y_min) * 0.5)
            
            mouth_x_min = max(0, mouth_x_min - mouth_padding_x)
            mouth_x_max = min(w, mouth_x_max + mouth_padding_x)
            mouth_y_min = max(0, mouth_y_min - mouth_padding_y)
            mouth_y_max = min(h, mouth_y_max + mouth_padding_y)
            
            return (x_min, y_min, x_max, y_max), (mouth_x_min, mouth_y_min, mouth_x_max, mouth_y_max)
        
        return (x_min, y_min, x_max, y_max), None
    
    def detect_face_opencv(self, frame):
        """使用OpenCV检测人脸和嘴部区域"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # 取最大的人脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # 估算嘴部区域（人脸下三分之一）
        mouth_y = y + int(h * 0.65)
        mouth_h = int(h * 0.35)
        mouth_x = x + int(w * 0.2)
        mouth_w = int(w * 0.6)
        
        return (x, y, x+w, y+h), (mouth_x, mouth_y, mouth_x+mouth_w, mouth_y+mouth_h)
    
    def extract_mouth_roi(self, frame, mouth_bbox, target_size=96):
        """提取并预处理嘴部区域
        
        Args:
            frame: 原始帧
            mouth_bbox: 嘴部边界框 (x_min, y_min, x_max, y_max)
            target_size: 目标尺寸
        
        Returns:
            mouth_roi: 处理后的嘴部区域 (96, 96)
        """
        if mouth_bbox is None:
            return None
        
        x_min, y_min, x_max, y_max = mouth_bbox
        
        # 确保边界框有效
        if x_max <= x_min or y_max <= y_min:
            return None
        
        # 裁剪嘴部区域
        mouth_roi = frame[y_min:y_max, x_min:x_max]
        
        if mouth_roi.size == 0:
            return None
        
        # 转换为灰度图
        if len(mouth_roi.shape) == 3:
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        
        # 调整大小到96x96
        mouth_roi = cv2.resize(mouth_roi, (target_size, target_size))
        
        return mouth_roi
    
    def preprocess_sequence(self, mouth_sequence):
        """预处理嘴部序列
        
        Args:
            mouth_sequence: 嘴部序列 (T, H, W)
        
        Returns:
            processed: 预处理后的tensor
        """
        # 应用预处理
        processed = self.preprocessing(mouth_sequence)
        
        # 转换为tensor
        processed = torch.from_numpy(processed).float()
        
        return processed
    
    def predict(self, mouth_sequence):
        """进行预测
        
        Args:
            mouth_sequence: 嘴部序列 (T, H, W)
        
        Returns:
            prediction: 预测的单词
            confidence: 置信度
            top5: Top-5预测结果
        """
        if len(mouth_sequence) < self.buffer_size:
            return None, 0.0, []
        
        # 预处理
        start_time = time.time()
        processed = self.preprocess_sequence(np.array(mouth_sequence))
        
        # 准备输入
        input_tensor = processed.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, T, H, W]
        lengths = torch.tensor([len(mouth_sequence)]).to(self.device)
        
        # 推理
        with torch.no_grad():
            logits = self.model(input_tensor, lengths=lengths)
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            # Top-5预测
            top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 构造结果
        prediction = self.labels[pred_idx.item()]
        top5 = [
            {'label': self.labels[idx.item()], 'probability': prob.item()}
            for idx, prob in zip(top5_indices[0], top5_probs[0])
        ]
        
        return prediction, confidence.item(), top5
    
    def draw_results(self, frame, face_bbox, mouth_bbox, fps):
        """在帧上绘制结果
        
        Args:
            frame: 原始帧
            face_bbox: 人脸边界框
            mouth_bbox: 嘴部边界框
            fps: 当前FPS
        """
        # 绘制人脸边界框
        if face_bbox:
            x_min, y_min, x_max, y_max = face_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 绘制嘴部边界框
        if mouth_bbox:
            x_min, y_min, x_max, y_max = mouth_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        # 显示预测结果
        y_offset = 30
        cv2.putText(frame, f"Prediction: {self.current_prediction}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Confidence: {self.confidence:.2f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示Top-5预测
        if self.top5_predictions:
            y_offset += 30
            cv2.putText(frame, "Top-5:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for i, pred in enumerate(self.top5_predictions[:5]):
                y_offset += 25
                text = f"{i+1}. {pred['label']}: {pred['probability']:.3f}"
                cv2.putText(frame, text, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示缓冲状态
        buffer_status = f"Buffer: {len(self.mouth_buffer)}/{self.buffer_size}"
        cv2.putText(frame, buffer_status, 
                   (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id=0, display=True):
        """运行实时检测
        
        Args:
            camera_id: 摄像头ID
            display: 是否显示结果
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {camera_id}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"摄像头已打开 (ID: {camera_id})")
        print("按 'q' 键退出")
        print("按 'r' 键重置缓冲")
        print("按 's' 键保存当前帧")
        
        frame_count = 0
        last_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("警告: 无法读取帧")
                    break
                
                frame_count += 1
                
                # 计算FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
                last_time = current_time
                self.fps_counter.append(fps)
                avg_fps = np.mean(self.fps_counter)
                
                # 检测人脸和嘴部
                if self.use_mediapipe:
                    result = self.detect_face_mediapipe(frame)
                else:
                    result = self.detect_face_opencv(frame)
                
                face_bbox = None
                mouth_bbox = None
                mouth_roi = None
                
                if result:
                    face_bbox, mouth_bbox = result
                    mouth_roi = self.extract_mouth_roi(frame, mouth_bbox)
                
                # 添加到缓冲
                if mouth_roi is not None:
                    self.mouth_buffer.append(mouth_roi)
                
                # 当缓冲满时进行预测
                if len(self.mouth_buffer) >= self.buffer_size:
                    prediction, confidence, top5 = self.predict(list(self.mouth_buffer))
                    
                    if prediction:
                        self.current_prediction = prediction
                        self.confidence = confidence
                        self.top5_predictions = top5
                
                # 绘制结果
                if display:
                    result_frame = self.draw_results(frame, face_bbox, mouth_bbox, avg_fps)
                    cv2.imshow('Real-time Lipreading', result_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("退出...")
                    break
                elif key == ord('r'):
                    # 重置缓冲
                    self.mouth_buffer.clear()
                    self.current_prediction = ""
                    self.confidence = 0.0
                    self.top5_predictions = []
                    print("缓冲已重置")
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"帧已保存: {filename}")
        
        except KeyboardInterrupt:
            print("\n检测被中断")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            # 打印统计信息
            if self.inference_times:
                avg_inference = np.mean(self.inference_times)
                print(f"\n统计信息:")
                print(f"  平均FPS: {avg_fps:.1f}")
                print(f"  平均推理时间: {avg_inference*1000:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description='实时摄像头唇语检测')
    
    # 模型参数
    parser.add_argument('--config-path', 
                       default='./configs/lrw_resnet18_dctcn.json',
                       help='模型配置文件路径')
    parser.add_argument('--model-path',
                       default='./models/lrw_resnet18_dctcn_video.pth',
                       help='预训练模型路径')
    parser.add_argument('--label-path',
                       default='./labels/500WordsSortedList.txt',
                       help='标签文件路径')
    
    # 检测参数
    parser.add_argument('--camera-id', type=int, default=0,
                       help='摄像头ID (默认: 0)')
    parser.add_argument('--buffer-size', type=int, default=29,
                       help='帧缓冲大小 (默认: 29)')
    parser.add_argument('--use-mediapipe', action='store_true', default=True,
                       help='使用MediaPipe进行人脸检测 (默认: True)')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示检测结果')
    
    # 设备参数
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='运行设备: auto(自动检测), cuda(GPU), cpu (默认: auto)')
    
    args = parser.parse_args()
    
    # 创建实时检测器
    detector = RealTimeLipreading(
        config_path=args.config_path,
        model_path=args.model_path,
        label_path=args.label_path,
        buffer_size=args.buffer_size,
        use_mediapipe=args.use_mediapipe,
        device=args.device
    )
    
    # 运行检测
    detector.run(camera_id=args.camera_id, display=not args.no_display)


if __name__ == '__main__':
    main()
