#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据准备脚本
用于创建示例测试数据或检查现有数据
"""

import os
import sys
import numpy as np
import argparse

def create_sample_npz(output_path, num_frames=29, height=96, width=96):
    """创建示例NPZ文件
    
    Args:
        output_path: 输出文件路径
        num_frames: 帧数
        height: 高度
        width: 宽度
    """
    # 创建随机数据（模拟嘴部区域）
    # 实际数据应该是真实的人脸嘴部区域
    data = np.random.randint(0, 255, (num_frames, height, width), dtype=np.uint8)
    
    # 添加一些结构（模拟嘴部）
    center_y, center_x = height // 2, width // 2
    for t in range(num_frames):
        # 模拟嘴部运动
        mouth_height = 10 + 5 * np.sin(2 * np.pi * t / num_frames)
        mouth_width = 30
        
        y_start = int(center_y - mouth_height // 2)
        y_end = int(center_y + mouth_height // 2)
        x_start = int(center_x - mouth_width // 2)
        x_end = int(center_x + mouth_width // 2)
        
        # 填充嘴部区域（较暗）
        data[t, y_start:y_end, x_start:x_end] = 100
    
    # 保存为NPZ格式
    np.savez(output_path, data=data)
    print(f"Created: {output_path}")


def check_data_directory(data_dir):
    """检查数据目录
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        stats: 统计信息字典
    """
    stats = {
        'total_files': 0,
        'total_words': 0,
        'words': {}
    }
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return stats
    
    # 遍历所有单词目录
    for word in os.listdir(data_dir):
        word_dir = os.path.join(data_dir, word)
        if not os.path.isdir(word_dir):
            continue
        
        # 检查test目录
        test_dir = os.path.join(word_dir, 'test')
        if os.path.exists(test_dir):
            npz_files = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
            if npz_files:
                stats['words'][word] = len(npz_files)
                stats['total_files'] += len(npz_files)
                stats['total_words'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare test data for lipreading model')
    
    parser.add_argument('--mode', default='check', choices=['check', 'create', 'info'],
                       help='Operation mode: check existing data, create sample data, or show info')
    parser.add_argument('--data-dir', default='./test_data/processed',
                       help='Data directory path')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of sample files to create per word')
    parser.add_argument('--words', nargs='+', default=['ABOUT', 'ABSOLUTELY', 'ABILITY'],
                       help='Words to create sample data for')
    
    args = parser.parse_args()
    
    if args.mode == 'check':
        # 检查现有数据
        print("="*60)
        print("检查测试数据")
        print("="*60)
        print(f"\n数据目录: {args.data_dir}\n")
        
        stats = check_data_directory(args.data_dir)
        
        if stats['total_files'] == 0:
            print("未找到测试数据")
            print("\n建议:")
            print("1. 使用 --mode create 创建示例数据")
            print("2. 或参考 test_data/README.md 准备真实数据")
        else:
            print(f"找到 {stats['total_words']} 个单词类别")
            print(f"总共 {stats['total_files']} 个测试视频\n")
            
            print("各单词数据量:")
            for word, count in sorted(stats['words'].items()):
                print(f"  {word}: {count} 个视频")
    
    elif args.mode == 'create':
        # 创建示例数据
        print("="*60)
        print("创建示例测试数据")
        print("="*60)
        print(f"\n目标目录: {args.data_dir}")
        print(f"单词列表: {args.words}")
        print(f"每个单词创建 {args.num_samples} 个样本\n")
        
        for word in args.words:
            # 创建目录
            word_dir = os.path.join(args.data_dir, word, 'test')
            os.makedirs(word_dir, exist_ok=True)
            
            # 创建样本文件
            for i in range(args.num_samples):
                output_path = os.path.join(word_dir, f'sample_{i+1}.npz')
                create_sample_npz(output_path)
        
        print(f"\n完成! 创建了 {len(args.words) * args.num_samples} 个示例文件")
        print("\n注意: 这些是随机生成的示例数据，仅用于测试代码流程")
        print("实际测试请使用真实的人脸嘴部视频数据")
    
    elif args.mode == 'info':
        # 显示信息
        print("="*60)
        print("测试数据信息")
        print("="*60)
        print("""
数据格式要求:
- 文件格式: NPZ (NumPy压缩格式)
- 数据字段: 'data'
- 数据形状: (T, H, W)
  - T: 时间帧数 (建议 25-30 帧)
  - H: 高度 = 96 像素
  - W: 宽度 = 96 像素
- 数据类型: uint8 (灰度图像)

目录结构:
test_data/processed/
├── WORD1/
│   └── test/
│       ├── video1.npz
│       └── video2.npz
├── WORD2/
│   └── test/
│       └── video1.npz
└── ...

获取真实数据:
1. LRW数据集: http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html
2. 自定义视频: 使用 preprocessing/convert_mp4_to_npz.py 转换
        """)


if __name__ == '__main__':
    main()
