#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet18-DCTCN视频模型测试脚本
支持单视频测试和批量测试
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(__file__))
from lipreading.model import Lipreading
from lipreading.preprocess import *
from lipreading.dataloaders import get_preprocessing_pipelines


def setup_device(device='auto'):
    """设置运行设备
    
    Args:
        device: 'auto', 'cuda', 或 'cpu'
    
    Returns:
        torch.device对象
    """
    if device == 'auto':
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


def load_model(config_path, model_path, num_classes=500, device='auto'):
    """加载预训练模型
    
    Args:
        config_path: 模型配置文件路径
        model_path: 预训练权重路径
        num_classes: 类别数量
        device: 运行设备
    
    Returns:
        model: 加载好的模型
    """
    # 设置设备
    device = setup_device(device)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 构建DenseTCN配置
    densetcn_options = {
        'block_config': config['densetcn_block_config'],
        'growth_rate_set': config['densetcn_growth_rate_set'],
        'reduced_size': config['densetcn_reduced_size'],
        'kernel_size_set': config['densetcn_kernel_size_set'],
        'dilation_size_set': config['densetcn_dilation_size_set'],
        'squeeze_excitation': config['densetcn_se'],
        'dropout': config['densetcn_dropout'],
    }
    
    # 创建模型
    model = Lipreading(
        modality='video',
        num_classes=num_classes,
        backbone_type=config['backbone_type'],
        relu_type=config['relu_type'],
        width_mult=config['width_mult'],
        densetcn_options=densetcn_options,
        use_boundary=config.get('use_boundary', False)
    )
    
    # 加载权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model path {model_path} does not exist, using random weights")
    
    # 移动模型到指定设备
    model = model.to(device)
    model.eval()
    
    # CPU优化
    if device.type == 'cpu':
        torch.set_num_threads(4)
        print("CPU优化: 使用4线程")
    
    return model, device


def load_labels(label_path):
    """加载标签列表
    
    Args:
        label_path: 标签文件路径
    
    Returns:
        labels: 标签列表
    """
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} not found")
        return [f"word_{i}" for i in range(500)]
    
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def preprocess_video(video_data):
    """预处理视频数据
    
    Args:
        video_data: numpy数组，shape=(T, H, W)
    
    Returns:
        processed_data: 预处理后的数据
        length: 序列长度
    """
    # 获取预处理管道
    preprocessing = get_preprocessing_pipelines('video')
    preprocess_pipeline = preprocessing['val']  # 使用验证集预处理
    
    # 应用预处理
    processed_data = preprocess_pipeline(video_data)
    
    # 转换为tensor
    processed_data = torch.from_numpy(processed_data).float()
    length = processed_data.shape[0]
    
    return processed_data, length


def test_single_video(model, video_path, labels, use_boundary=False, device=None):
    """测试单个视频
    
    Args:
        model: 模型
        video_path: 视频文件路径（.npz或.npy）
        labels: 标签列表
        use_boundary: 是否使用边界信息
        device: 运行设备
    
    Returns:
        result: 预测结果字典
    """
    # 加载数据
    if video_path.endswith('.npz'):
        video_data = np.load(video_path)['data']
    elif video_path.endswith('.npy'):
        video_data = np.load(video_path)
    else:
        raise ValueError(f"Unsupported file format: {video_path}")
    
    # 预处理
    processed_data, length = preprocess_video(video_data)
    
    # 准备输入
    input_tensor = processed_data.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T, H, W]
    lengths = torch.tensor([length]).to(device)
    
    # 推理
    with torch.no_grad():
        if use_boundary:
            # 如果使用边界，需要提供边界信息
            boundaries = torch.zeros(1, length).to(device)
            logits = model(input_tensor, lengths=lengths, boundaries=boundaries)
        else:
            logits = model(input_tensor, lengths=lengths)
        
        # 获取预测结果
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        
        # 获取top-5预测
        top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
    
    # 构造结果
    result = {
        'video_path': video_path,
        'predicted_label': labels[pred_idx.item()],
        'confidence': confidence.item(),
        'top5_predictions': [
            {'label': labels[idx.item()], 'probability': prob.item()}
            for idx, prob in zip(top5_indices[0], top5_probs[0])
        ],
        'video_shape': video_data.shape,
        'num_frames': length
    }
    
    return result


def test_batch_videos(model, data_dir, labels, use_boundary=False, split='test', device=None):
    """批量测试视频
    
    Args:
        model: 模型
        data_dir: 数据目录
        labels: 标签列表
        use_boundary: 是否使用边界信息
        split: 数据集划分 ('train', 'val', 'test')
        device: 运行设备
    
    Returns:
        results: 测试结果列表
        accuracy: 准确率
    """
    results = []
    correct = 0
    total = 0
    
    # 遍历所有单词类别
    for word_idx, word in enumerate(tqdm(labels, desc="Testing words")):
        word_dir = os.path.join(data_dir, word, split)
        
        if not os.path.exists(word_dir):
            continue
        
        # 遍历该单词的所有视频
        video_files = [f for f in os.listdir(word_dir) if f.endswith('.npz')]
        
        for video_file in video_files:
            video_path = os.path.join(word_dir, video_file)
            
            try:
                result = test_single_video(model, video_path, labels, use_boundary, device)
                result['true_label'] = word
                result['true_label_idx'] = word_idx
                results.append(result)
                
                # 计算准确率
                if result['predicted_label'] == word:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
    
    accuracy = correct / total if total > 0 else 0
    
    return results, accuracy


def print_result(result):
    """打印单个结果"""
    print("\n" + "="*60)
    print(f"Video: {result['video_path']}")
    print(f"Shape: {result['video_shape']}, Frames: {result['num_frames']}")
    print(f"\nPredicted: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
    print("\nTop-5 Predictions:")
    for i, pred in enumerate(result['top5_predictions'], 1):
        print(f"  {i}. {pred['label']}: {pred['probability']:.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Test ResNet18-DCTCN video model')
    
    # 模型相关
    parser.add_argument('--config-path', 
                       default='./configs/lrw_resnet18_dctcn.json',
                       help='Model config file path')
    parser.add_argument('--model-path',
                       default='./models/lrw_resnet18_dctcn_video.pth',
                       help='Pretrained model path')
    parser.add_argument('--num-classes', type=int, default=500,
                       help='Number of classes')
    
    # 数据相关
    parser.add_argument('--data-dir', default='./test_data/processed',
                       help='Data directory for batch testing')
    parser.add_argument('--video-path', default=None,
                       help='Single video path for testing (overrides batch mode)')
    parser.add_argument('--label-path', default='./labels/500WordsSortedList.txt',
                       help='Label file path')
    
    # 测试模式
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to test')
    parser.add_argument('--use-boundary', action='store_true',
                       help='Use boundary information')
    
    # 输出
    parser.add_argument('--output', default=None,
                       help='Output file for results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results')
    
    # 设备参数
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='运行设备: auto(自动检测), cuda(GPU), cpu (默认: auto)')
    
    args = parser.parse_args()
    
    # 加载模型
    print("Loading model...")
    model, device = load_model(args.config_path, args.model_path, args.num_classes, args.device)
    
    # 加载标签
    labels = load_labels(args.label_path)
    print(f"Loaded {len(labels)} labels")
    
    # 测试
    if args.video_path:
        # 单视频测试模式
        print(f"\nTesting single video: {args.video_path}")
        result = test_single_video(model, args.video_path, labels, args.use_boundary, device)
        
        if args.verbose:
            print_result(result)
        else:
            print(f"\nPredicted: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
        
        results = [result]
        accuracy = None
        
    else:
        # 批量测试模式
        print(f"\nBatch testing on {args.split} set...")
        results, accuracy = test_batch_videos(
            model, args.data_dir, labels, args.use_boundary, args.split, device
        )
        
        print(f"\nTest Results:")
        print(f"  Total videos: {len(results)}")
        print(f"  Accuracy: {accuracy:.4f} ({int(accuracy*len(results))}/{len(results)})")
    
    # 保存结果
    if args.output:
        output_data = {
            'results': results,
            'accuracy': accuracy
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
