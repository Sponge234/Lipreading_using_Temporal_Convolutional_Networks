# ResNet18-DCTCN 视频模型测试指南

本指南介绍如何测试 ResNet18-DCTCN 唇读模型，包括如何处理原始 MP4 视频数据。

## 目录结构

```
test_data/
├── raw_videos/          # 存放原始MP4视频
├── processed/           # 存放处理后的NPZ文件
│   ├── WORD1/          # 单词类别目录
│   │   └── test/       # 测试集
│   │       ├── video1.npz
│   │       └── video2.npz
│   └── WORD2/
│       └── test/
└── README.md           # 本文档
```

## 数据格式要求

### 输入视频要求
- **格式**: MP4
- **内容**: 人脸视频，包含清晰的嘴部区域
- **帧率**: 建议 25 FPS
- **分辨率**: 建议 720p 或更高
- **时长**: 建议 1-2 秒（包含完整的单词发音）

### 处理后的数据格式
- **格式**: NPZ (NumPy压缩格式)
- **数据字段**: `data`
- **数据形状**: `(T, H, W)` 
  - T: 时间帧数（可变）
  - H: 高度 = 96 像素
  - W: 宽度 = 96 像素
- **数据类型**: uint8 (灰度图像)

## 使用步骤

### 步骤1: 准备原始视频

将原始 MP4 视频放入 `test_data/raw_videos/` 目录：

```bash
# 示例：复制视频到测试目录
cp /path/to/your/videos/*.mp4 test_data/raw_videos/
```

### 步骤2: 转换视频格式

使用转换脚本将 MP4 视频转换为 NPZ 格式：

#### 方式1: 使用预计算的人脸关键点（推荐）

如果你有预计算的人脸关键点文件（.pkl格式）：

```bash
python preprocessing/convert_mp4_to_npz.py \
    --video-direc test_data/raw_videos \
    --landmark-direc /path/to/landmarks \
    --output-direc test_data/processed \
    --convert-gray
```

#### 方式2: 自动检测人脸关键点

如果没有预计算的关键点，可以使用自动检测：

```bash
# 使用 MediaPipe（推荐，更准确）
python preprocessing/convert_mp4_to_npz.py \
    --video-direc test_data/raw_videos \
    --output-direc test_data/processed \
    --auto-detect \
    --detector mediapipe

# 或使用 dlib
python preprocessing/convert_mp4_to_npz.py \
    --video-direc test_data/raw_videos \
    --output-direc test_data/processed \
    --auto-detect \
    --detector dlib
```

**注意**: 自动检测需要安装额外的依赖：
```bash
# MediaPipe
pip install mediapipe

# 或 dlib
pip install dlib
# 还需要下载关键点模型文件
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

### 步骤3: 组织测试数据

将处理后的 NPZ 文件按照单词类别组织：

```bash
# 示例目录结构
test_data/processed/
├── ABOUT/
│   └── test/
│       ├── video1.npz
│       └── video2.npz
├── ABSOLUTELY/
│   └── test/
│       └── video1.npz
└── ...
```

### 步骤4: 运行测试

#### 测试单个视频

```bash
python test_resnet18_dctcn.py \
    --config-path ./configs/lrw_resnet18_dctcn.json \
    --model-path ./models/lrw_resnet18_dctcn_video.pth \
    --video-path test_data/processed/ABOUT/test/video1.npz \
    --verbose
```

#### 批量测试

```bash
python test_resnet18_dctcn.py \
    --config-path ./configs/lrw_resnet18_dctcn.json \
    --model-path ./models/lrw_resnet18_dctcn_video.pth \
    --data-dir test_data/processed \
    --split test \
    --output results.json
```

## 测试脚本参数说明

### 模型参数
- `--config-path`: 模型配置文件路径（默认: `./configs/lrw_resnet18_dctcn.json`）
- `--model-path`: 预训练模型路径（默认: `./models/lrw_resnet18_dctcn_video.pth`）
- `--num-classes`: 类别数量（默认: 500）

### 数据参数
- `--data-dir`: 批量测试的数据目录
- `--video-path`: 单视频测试路径（设置后将忽略批量测试）
- `--label-path`: 标签文件路径（默认: `./labels/500WordsSortedList.txt`）
- `--split`: 测试的数据集划分（train/val/test，默认: test）

### 输出参数
- `--output`: 结果输出文件（JSON格式）
- `--verbose`: 打印详细结果

## 输出示例

### 单视频测试输出

```
============================================================
Video: test_data/processed/ABOUT/test/video1.npz
Shape: (29, 96, 96), Frames: 29

Predicted: ABOUT (confidence: 0.9234)

Top-5 Predictions:
  1. ABOUT: 0.9234
  2. ABSOLUTELY: 0.0312
  3. ABILITY: 0.0187
  4. ABLE: 0.0123
  5. ABROAD: 0.0089
============================================================
```

### 批量测试输出

```
Test Results:
  Total videos: 1000
  Accuracy: 0.8960 (896/1000)

Results saved to results.json
```

## 获取测试数据

### LRW 数据集

本模型在 LRW (Lip Reading in the Wild) 数据集上训练。如需获取数据：

1. **官方申请**: 访问 [LRW数据集官网](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) 申请下载
2. **数据集包含**:
   - 500 个单词类别
   - 约 160,000 个视频片段
   - 训练/验证/测试集划分

### 创建自定义测试数据

如果使用自己的视频数据：

1. 确保视频包含清晰的人脸和嘴部区域
2. 视频时长建议 1-2 秒，包含完整单词发音
3. 使用提供的转换脚本处理视频
4. 按照单词类别组织数据目录

## 模型性能

在 LRW 测试集上的性能：
- **准确率**: 89.6%
- **Top-5 准确率**: 98.2%

## 常见问题

### Q1: 转换视频时提示找不到人脸

**解决方案**:
- 确保视频中有清晰的人脸
- 尝试使用 MediaPipe 检测器（更鲁棒）
- 检查视频质量，确保光线充足

### Q2: 测试时出现维度错误

**解决方案**:
- 检查 NPZ 文件的数据形状是否为 `(T, 96, 96)`
- 确保数据类型为 uint8
- 验证预处理是否正确应用

### Q3: 模型加载失败

**解决方案**:
- 检查模型文件路径是否正确
- 确保模型文件完整（约 201MB）
- 验证配置文件与模型匹配

### Q4: GPU 内存不足

**解决方案**:
- 减小 batch size
- 使用单视频测试模式
- 考虑使用 CPU 测试（速度较慢）

## 依赖安装

```bash
# 基础依赖
pip install torch torchvision
pip install numpy opencv-python tqdm

# 视频处理
pip install av  # 或 ffmpeg-python

# 人脸检测（可选）
pip install mediapipe  # 推荐
# 或
pip install dlib
```

## 相关文件

- **模型定义**: `lipreading/model.py`
- **预处理代码**: `lipreading/preprocess.py`
- **数据加载器**: `lipreading/dataloaders.py`
- **转换脚本**: `preprocessing/convert_mp4_to_npz.py`
- **测试脚本**: `test_resnet18_dctcn.py`

## 参考资料

- [原始论文](https://arxiv.org/abs/2001.08702): "Lipreading using Temporal Convolutional Networks"
- [项目仓库](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)
- [LRW数据集](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
