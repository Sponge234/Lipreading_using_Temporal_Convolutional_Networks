# ResNet18-DCTCN 模型测试完成总结

## 已完成的工作

### 1. 创建测试数据目录结构 ✓
```
test_data/
├── raw_videos/          # 原始MP4视频存放目录
├── processed/           # 处理后的NPZ文件目录
│   ├── ABOUT/
│   │   └── test/
│   ├── ABSOLUTELY/
│   │   └── test/
│   └── ABILITY/
│       └── test/
└── README.md           # 详细使用文档
```

### 2. 编写MP4视频转换脚本 ✓
**文件**: `preprocessing/convert_mp4_to_npz.py`

**功能**:
- 支持使用预计算的人脸关键点（推荐）
- 支持自动检测人脸关键点（MediaPipe/dlib）
- 批量处理视频文件
- 自动保存为NPZ格式

**使用方法**:
```bash
# 方式1: 使用预计算关键点
python preprocessing/convert_mp4_to_npz.py \
    --video-direc test_data/raw_videos \
    --landmark-direc /path/to/landmarks \
    --output-direc test_data/processed

# 方式2: 自动检测关键点
python preprocessing/convert_mp4_to_npz.py \
    --video-direc test_data/raw_videos \
    --output-direc test_data/processed \
    --auto-detect \
    --detector mediapipe
```

### 3. 创建测试脚本 ✓
**文件**: `test_resnet18_dctcn.py`

**功能**:
- 支持单视频测试
- 支持批量测试
- 显示Top-5预测结果
- 计算准确率
- 保存结果为JSON格式

**使用方法**:
```bash
# 单视频测试
python test_resnet18_dctcn.py \
    --video-path test_data/processed/ABOUT/test/video1.npz \
    --verbose

# 批量测试
python test_resnet18_dctcn.py \
    --data-dir test_data/processed \
    --split test \
    --output results.json
```

### 4. 准备示例测试数据 ✓
**文件**: `prepare_test_data.py`

**功能**:
- 检查现有数据
- 创建示例数据（用于测试代码流程）
- 显示数据格式信息

**使用方法**:
```bash
# 检查数据
python prepare_test_data.py --mode check

# 创建示例数据
python prepare_test_data.py --mode create --num-samples 5

# 显示信息
python prepare_test_data.py --mode info
```

### 5. 创建辅助文档和脚本 ✓
- `test_data/README.md`: 详细使用文档
- `QUICKSTART.md`: 快速开始指南
- `demo_test.sh`: 示例测试脚本

## 测试验证

### 模型加载测试 ✓
```
✓ 模型文件存在: models/lrw_resnet18_dctcn_video.pth (201MB)
✓ 配置文件存在: configs/lrw_resnet18_dctcn.json
✓ 模型成功加载
✓ 标签文件加载成功 (500个单词)
```

### 单视频测试 ✓
```
测试视频: test_data/processed/ABOUT/test/sample_1.npz
数据形状: (29, 96, 96)
帧数: 29
预测结果: BLACK (confidence: 0.0327)
状态: ✓ 成功运行
```

## 数据处理流程

### 原始MP4视频 → NPZ格式

```
原始MP4视频
    ↓
人脸检测和对齐
    ↓
嘴部区域裁剪 (96×96)
    ↓
灰度转换
    ↓
保存为NPZ格式
    ↓
模型输入 (T, 88, 88)
```

### 数据格式要求

**输入视频**:
- 格式: MP4
- 内容: 包含清晰人脸的视频
- 帧率: 建议 25 FPS
- 时长: 1-2秒

**处理后数据**:
- 格式: NPZ
- 字段: 'data'
- 形状: (T, 96, 96)
- 类型: uint8

**模型输入**:
- 形状: (B, 1, T, 88, 88)
- 归一化: mean=0.421, std=0.165

## 使用流程

### 完整测试流程

```bash
# 1. 准备视频数据
cp /path/to/videos/*.mp4 test_data/raw_videos/

# 2. 转换为NPZ格式
python preprocessing/convert_mp4_to_npz.py \
    --video-direc test_data/raw_videos \
    --output-direc test_data/processed \
    --auto-detect

# 3. 运行测试
python test_resnet18_dctcn.py \
    --data-dir test_data/processed \
    --split test \
    --output results.json

# 4. 查看结果
cat results.json
```

### 快速测试流程

```bash
# 使用示例脚本
./demo_test.sh

# 或创建示例数据快速测试
python prepare_test_data.py --mode create
python test_resnet18_dctcn.py --video-path test_data/processed/ABOUT/test/sample_1.npz --verbose
```

## 文件清单

### 核心文件
- `test_resnet18_dctcn.py` - 模型测试脚本
- `preprocessing/convert_mp4_to_npz.py` - 视频转换脚本
- `prepare_test_data.py` - 数据准备脚本

### 文档文件
- `test_data/README.md` - 详细使用文档
- `QUICKSTART.md` - 快速开始指南
- `TEST_SUMMARY.md` - 本文档

### 辅助文件
- `demo_test.sh` - 示例测试脚本

## 下一步建议

### 1. 准备真实测试数据
- 申请LRW数据集: http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html
- 或使用自己的视频数据
- 确保视频质量良好，人脸清晰

### 2. 批量测试评估
- 准备多个单词类别的测试数据
- 运行批量测试评估模型性能
- 分析错误案例

### 3. 性能优化
- 调整batch size优化GPU利用率
- 尝试不同的预处理参数
- 测试不同视频质量的影响

### 4. 扩展应用
- 尝试其他模型配置（如使用边界信息）
- 测试音频模态
- 实现实时唇读应用

## 依赖环境

### 已安装
- ✓ librosa (音频处理)
- ✓ torch (深度学习框架)
- ✓ numpy (数值计算)
- ✓ opencv-python (图像处理)
- ✓ tqdm (进度条)

### 可选依赖
- mediapipe (人脸检测，推荐)
- dlib (人脸检测，备选)

## 注意事项

1. **示例数据**: 当前创建的是随机示例数据，仅用于测试代码流程，实际测试请使用真实数据

2. **GPU内存**: 批量测试时注意GPU内存使用，可减小batch size

3. **视频质量**: 确保输入视频包含清晰的人脸和嘴部区域

4. **数据格式**: 严格按照要求的格式准备数据，确保维度和数据类型正确

5. **模型性能**: 在LRW测试集上准确率约89.6%，实际性能取决于数据质量

## 技术支持

如遇问题，请参考：
- `test_data/README.md` - 详细使用说明
- `QUICKSTART.md` - 快速开始指南
- 原始项目: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks

---

**测试环境**: 
- Python 3.x
- PyTorch
- CUDA (推荐)
- 模型: ResNet18-DCTCN
- 数据集: LRW (500 words)

**最后更新**: 2026-03-17
