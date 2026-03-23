# 唇语识别系统使用指南

本项目基于 ResNet18-DCTCN 模型实现实时唇语识别，可识别 500 个英语单词。

## 目录

- [环境配置](#环境配置)
- [模型下载](#模型下载)
- [实时识别](#实时识别)
- [离线测试](#离线测试)
- [参数说明](#参数说明)
- [常见问题](#常见问题)

---

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install mediapipe  # 用于实时人脸检测
```

### 2. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## 模型下载

### 自动下载（推荐）

首次运行时会自动下载模型（约 45MB）：

```bash
python realtime_lipreading.py
```

### 手动下载

如果自动下载失败，请手动下载：

```bash
# 创建模型目录
mkdir -p models

# 下载模型文件
wget -O models/lrw_resnet18_dctcn_video.pth \
  https://github.com/mpc001/Hubert_avsr/releases/download/v0.1/lrw_resnet18_dctcn_video.pth
```

---

## 实时识别

### 快速启动

```bash
# 使用默认设置
python realtime_lipreading.py

# 或使用启动脚本
bash start_realtime.sh
```

### 自定义参数

```bash
# 指定摄像头和历史记录时长
python realtime_lipreading.py \
  --camera-id 0 \
  --history-duration 10.0 \
  --min-confidence 0.4

# 使用 CPU 运行
python realtime_lipreading.py --device cpu

# 使用 GPU 运行
python realtime_lipreading.py --device cuda
```

### 交互控制

运行时可通过键盘控制：

- **q** - 退出程序
- **r** - 重置缓冲和历史记录
- **s** - 保存当前帧

### 界面说明

实时界面显示以下信息：

- **Current**: 当前识别的单词
- **Confidence**: 识别置信度
- **Sentence**: 历史识别组合的句子
- **Top-5**: 前5个候选单词
- **FPS**: 实时帧率
- **Buffer**: 帧缓冲状态
- **History**: 历史记录单词数

---

## 离线测试

### 测试视频文件

```bash
python test_resnet18_dctcn.py \
  --config-path ./configs/lrw_resnet18_dctcn.json \
  --model-path ./models/lrw_resnet18_dctcn_video.pth \
  --label-path ./labels/500WordsSortedList.txt \
  --video-path ./test_data/sample.mp4
```

### 批量测试

```bash
python test_resnet18_dctcn.py \
  --video-dir ./test_data/ \
  --output results.json
```

---

## 参数说明

### 实时识别参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config-path` | `./configs/lrw_resnet18_dctcn.json` | 模型配置文件 |
| `--model-path` | `./models/lrw_resnet18_dctcn_video.pth` | 模型权重文件 |
| `--label-path` | `./labels/500WordsSortedList.txt` | 标签文件 |
| `--camera-id` | `0` | 摄像头ID |
| `--buffer-size` | `29` | 帧缓冲大小 |
| `--use-mediapipe` | `True` | 使用MediaPipe检测 |
| `--device` | `auto` | 运行设备 (auto/cuda/cpu) |
| `--history-duration` | `5.0` | 历史记录保留时长(秒) |
| `--min-confidence` | `0.3` | 最小置信度阈值 |

### 历史记录功能

系统会将识别的单词保留在屏幕上，形成连续句子效果：

- **保留时长**: 通过 `--history-duration` 设置（默认5秒）
- **置信度过滤**: 低于 `--min-confidence` 的预测不记录
- **防重复**: 短时间内不重复记录相同单词
- **自动清理**: 超过保留时长的记录自动删除

---

## 常见问题

### 1. 摄像头无法打开

```bash
# 检查摄像头设备
ls /dev/video*

# 尝试不同的摄像头ID
python realtime_lipreading.py --camera-id 1
```

### 2. GPU 内存不足

```bash
# 使用 CPU 运行
python realtime_lipreading.py --device cpu
```

### 3. 识别准确率低

- 确保光线充足
- 正对摄像头，保持面部清晰
- 说话时嘴部动作明显
- 调整置信度阈值：`--min-confidence 0.5`

### 4. FPS 过低

- 使用 GPU：`--device cuda`
- 降低帧缓冲：`--buffer-size 20`
- 关闭其他占用GPU的程序

### 5. MediaPipe 安装失败

```bash
# 使用 OpenCV 检测器替代
python realtime_lipreading.py --use-mediapipe False
```

---

## 性能参考

| 设备 | FPS | 推理时间 |
|------|-----|----------|
| GPU (RTX 3090) | 25-30 | ~30ms |
| GPU (GTX 1080) | 20-25 | ~40ms |
| CPU (i7-9700K) | 8-12 | ~80ms |
| CPU (i5-8250U) | 5-8 | ~150ms |

---

## 项目结构

```
.
├── realtime_lipreading.py    # 实时识别主程序
├── test_resnet18_dctcn.py    # 离线测试程序
├── main.py                   # 训练主程序
├── start_realtime.sh         # 启动脚本
├── configs/                  # 模型配置
├── models/                   # 模型权重
├── labels/                   # 标签文件
├── lipreading/               # 核心代码
│   ├── model.py              # 模型定义
│   ├── preprocess.py         # 预处理
│   └── dataloaders.py        # 数据加载
└── preprocessing/            # 预处理工具
```

---

## 更多信息

- 原始论文: [Lipreading using Temporal Convolutional Networks](https://arxiv.org/abs/1904.04612)
- 模型准确率: 89.6% (LRW测试集)
- 支持词汇: 500个英语单词
