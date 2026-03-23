#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试实时识别修复是否有效
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(__file__))
from lipreading.model import Lipreading
from lipreading.dataloaders import get_preprocessing_pipelines
import json


def test_preprocessing():
    """测试预处理流程"""
    print("=" * 50)
    print("测试预处理流程")
    print("=" * 50)

    # 获取预处理流程
    preprocessing = get_preprocessing_pipelines('video')['val']
    print(f"预处理流程: {preprocessing}")

    # 创建测试数据 (29帧，每帧88x88，模拟uint8图像)
    test_sequence = np.random.randint(0, 256, size=(29, 88, 88), dtype=np.uint8)
    print(f"\n输入数据形状: {test_sequence.shape}")
    print(f"输入数据类型: {test_sequence.dtype}")
    print(f"输入数据范围: [{test_sequence.min()}, {test_sequence.max()}]")

    # 应用预处理
    processed = preprocessing(test_sequence)
    print(f"\n预处理后形状: {processed.shape}")
    print(f"预处理后类型: {processed.dtype}")
    print(f"预处理后范围: [{processed.min():.4f}, {processed.max():.4f}]")
    print(f"预处理后均值: {processed.mean():.4f}")
    print(f"预处理后标准差: {processed.std():.4f}")

    return processed


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "=" * 50)
    print("测试模型前向传播")
    print("=" * 50)

    # 加载配置
    config_path = './configs/lrw_resnet18_dctcn.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"模型配置: {config}")

    # 创建模型
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

    model.eval()

    # 检查模型文件是否存在
    model_path = './models/lrw_resnet18_dctcn_video.pth'
    if os.path.exists(model_path):
        print(f"\n加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("模型权重加载成功")
    else:
        print(f"\n警告: 模型文件不存在 {model_path}")
        print("使用随机初始化的模型进行测试")

    # 创建测试输入
    # 输入形状应该是 [batch_size, channels, time, height, width]
    # 对于视频: [1, 1, 29, 88, 88]
    test_input = torch.randn(1, 1, 29, 88, 88)
    lengths = torch.tensor([29])

    print(f"\n测试输入形状: {test_input.shape}")
    print(f"输入数据范围: [{test_input.min():.4f}, {test_input.max():.4f}]")

    # 前向传播
    with torch.no_grad():
        logits = model(test_input, lengths=lengths)
        print(f"\n模型输出(logits)形状: {logits.shape}")
        print(f"Logits范围: [{logits.min():.4f}, {logits.max():.4f}]")

        # 应用softmax
        probs = torch.nn.functional.softmax(logits, dim=1)
        print(f"\nSoftmax后概率形状: {probs.shape}")
        print(f"概率范围: [{probs.min():.6f}, {probs.max():.6f}]")
        print(f"概率和: {probs.sum():.6f}")

        # 获取Top-5预测
        top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
        print(f"\nTop-5预测:")
        for i in range(5):
            print(f"  {i+1}. 索引: {top5_indices[0][i].item()}, 概率: {top5_probs[0][i].item():.6f}")


def test_realtime_preprocessing():
    """测试实时识别的预处理流程"""
    print("\n" + "=" * 50)
    print("测试实时识别预处理流程")
    print("=" * 50)

    from realtime_lipreading import RealTimeLipreading

    # 检查必要文件
    config_path = './configs/lrw_resnet18_dctcn.json'
    model_path = './models/lrw_resnet18_dctcn_video.pth'
    label_path = './labels/500WordsSortedList.txt'

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return

    if not os.path.exists(label_path):
        print(f"警告: 标签文件不存在 {label_path}")
        print("使用默认标签")

    # 创建检测器实例
    try:
        detector = RealTimeLipreading(
            config_path=config_path,
            model_path=model_path,
            label_path=label_path,
            buffer_size=29,
            use_mediapipe=False,  # 不使用MediaPipe以避免依赖问题
            device='cpu'
        )

        # 创建测试嘴部序列 (29帧，每帧88x88的灰度图)
        test_sequence = []
        for i in range(29):
            # 模拟嘴部区域，创建一些变化的模式
            frame = np.random.randint(50, 200, size=(88, 88), dtype=np.uint8)
            # 添加一些结构
            frame[30:60, 20:70] = np.random.randint(100, 180, size=(30, 50), dtype=np.uint8)
            test_sequence.append(frame)

        print(f"\n测试序列:")
        print(f"  帧数: {len(test_sequence)}")
        print(f"  每帧形状: {test_sequence[0].shape}")
        print(f"  每帧类型: {test_sequence[0].dtype}")

        # 测试预处理
        processed = detector.preprocess_sequence(test_sequence)
        print(f"\n预处理后:")
        print(f"  形状: {processed.shape}")
        print(f"  类型: {processed.dtype}")
        print(f"  范围: [{processed.min():.4f}, {processed.max():.4f}]")

        # 测试预测
        print(f"\n执行预测...")
        prediction, confidence, top5 = detector.predict(test_sequence)

        if prediction:
            print(f"\n预测结果:")
            print(f"  预测词: {prediction}")
            print(f"  置信度: {confidence:.6f}")
            print(f"  Top-5预测:")
            for i, pred in enumerate(top5):
                print(f"    {i+1}. {pred['label']}: {pred['probability']:.6f}")
        else:
            print("预测失败")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("开始测试实时识别修复...\n")

    # 测试1: 预处理流程
    test_preprocessing()

    # 测试2: 模型前向传播
    test_model_forward()

    # 测试3: 实时识别预处理和预测
    test_realtime_preprocessing()

    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
