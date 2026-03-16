# 实时摄像头唇语检测使用指南

## 功能介绍

实时摄像头唇语检测系统可以：
- 实时捕获摄像头视频流
- 自动检测人脸和嘴部区域
- 实时识别唇语并显示预测结果
- 显示Top-5预测结果和置信度
- 实时显示FPS和缓冲状态

## 系统要求

### 硬件要求
- 摄像头（USB摄像头或笔记本内置摄像头）
- NVIDIA GPU（推荐，用于加速推理）
- 至少4GB内存

### 软件要求
- Python 3.6+
- OpenCV
- PyTorch
- MediaPipe（推荐）或 OpenCV Haar分类器

## 安装依赖

```bash
# 基础依赖
pip install opencv-python torch numpy

# 推荐安装MediaPipe（更准确的人脸检测）
pip install mediapipe
```

## 使用方法

### 基本使用

```bash
# 使用默认参数运行
python realtime_lipreading.py

# 指定摄像头ID（如果有多个摄像头）
python realtime_lipreading.py --camera-id 1

# 使用OpenCV检测器（如果MediaPipe不可用）
python realtime_lipreading.py --use-mediapipe False
```

### 高级参数

```bash
# 自定义缓冲大小（影响识别延迟）
python realtime_lipreading.py --buffer-size 25

# 指定模型路径
python realtime_lipreading.py \
    --config-path ./configs/lrw_resnet18_dctcn.json \
    --model-path ./models/lrw_resnet18_dctcn_video.pth \
    --label-path ./labels/500WordsSortedList.txt
```

## 运行时控制

运行程序后，可以使用以下键盘控制：

- **q**: 退出程序
- **r**: 重置帧缓冲（清除当前累积的帧）
- **s**: 保存当前帧为图片

## 界面说明

运行时会显示实时视频窗口，包含以下信息：

```
┌─────────────────────────────────────┐
│ Prediction: ABOUT          FPS: 30  │
│ Confidence: 0.85          Buffer: 29/29│
│ Top-5:                              │
│ 1. ABOUT: 0.850                     │
│ 2. ABSOLUTELY: 0.050                │
│ 3. ABILITY: 0.030                   │
│ 4. ABLE: 0.020                      │
│ 5. ABROAD: 0.010                    │
│                                     │
│     ┌─────────────┐                 │
│     │   人脸框    │                 │
│     │  ┌─────┐   │                 │
│     │  │嘴部 │   │                 │
│     │  └─────┘   │                 │
│     └─────────────┘                 │
└─────────────────────────────────────┘
```

- **绿色框**: 人脸边界框
- **蓝色框**: 嘴部区域边界框
- **Prediction**: 当前预测的单词
- **Confidence**: 预测置信度（0-1）
- **Top-5**: 前5个最可能的预测结果
- **FPS**: 当前帧率
- **Buffer**: 帧缓冲状态（当前帧数/总帧数）

## 工作原理

### 1. 视频捕获
- 从摄像头实时捕获视频帧
- 默认分辨率：640x480
- 默认帧率：30 FPS

### 2. 人脸检测
- **MediaPipe模式**（推荐）：
  - 使用MediaPipe Face Mesh检测468个人脸关键点
  - 精确定位嘴部区域
  - 更鲁棒，适应不同角度和光照

- **OpenCV模式**（备选）：
  - 使用Haar级联分类器检测人脸
  - 根据人脸位置估算嘴部区域
  - 速度较快但精度较低

### 3. 嘴部区域提取
- 从检测到的嘴部边界框裁剪区域
- 转换为灰度图像
- 调整大小到96x96像素

### 4. 帧缓冲
- 维护一个固定大小的帧缓冲（默认29帧）
- 当缓冲满时触发模型推理
- 实现滑动窗口预测

### 5. 模型推理
- 使用ResNet18-DCTCN模型进行预测
- 输出500个单词类别的概率分布
- 显示Top-5预测结果

### 6. 结果显示
- 实时绘制检测结果
- 显示预测单词和置信度
- 显示性能统计信息

## 性能优化建议

### 1. 提高FPS
- 使用GPU加速（确保CUDA可用）
- 减小输入分辨率
- 使用更小的buffer_size

### 2. 提高准确率
- 确保良好的光照条件
- 保持正脸朝向摄像头
- 清晰发音，嘴部动作明显
- 使用MediaPipe检测器

### 3. 降低延迟
- 减小buffer_size（但可能影响准确率）
- 使用更快的GPU

## 常见问题

### Q1: 无法打开摄像头

**解决方案**:
```bash
# 检查摄像头设备
ls /dev/video*

# 尝试不同的摄像头ID
python realtime_lipreading.py --camera-id 0
python realtime_lipreading.py --camera-id 1
```

### Q2: FPS很低

**解决方案**:
- 确保使用GPU：`nvidia-smi` 检查GPU状态
- 检查CUDA是否正确安装
- 关闭其他占用GPU的程序

### Q3: 检测不到人脸

**解决方案**:
- 确保光线充足
- 正脸朝向摄像头
- 尝试调整摄像头角度
- 使用MediaPipe检测器（更鲁棒）

### Q4: 预测结果不准确

**原因**:
- 模型在LRW数据集上训练，主要识别英语单词
- 需要清晰的嘴部动作
- 建议使用正脸、良好光照

**建议**:
- 发音时嘴部动作要清晰明显
- 保持稳定的头部位置
- 确保嘴部区域被正确检测

### Q5: MediaPipe安装失败

**解决方案**:
```bash
# 使用OpenCV检测器作为备选
python realtime_lipreading.py --use-mediapipe False

# 或尝试安装MediaPipe
pip install mediapipe --upgrade
```

## 使用场景

### 1. 演示和测试
- 展示唇语识别技术
- 测试模型性能
- 验证系统功能

### 2. 辅助交流
- 为听障人士提供辅助
- 嘈杂环境下的语音识别补充

### 3. 研究开发
- 收集唇语数据
- 测试新算法
- 性能基准测试

## 技术参数

### 模型参数
- **架构**: ResNet18 + DenseTCN
- **输入尺寸**: 88x88像素（从96x96裁剪）
- **帧数**: 29帧（可配置）
- **类别数**: 500个英语单词
- **准确率**: ~89.6% (LRW测试集)

### 检测参数
- **人脸检测**: MediaPipe Face Mesh / OpenCV Haar
- **嘴部ROI**: 96x96像素
- **帧缓冲**: 29帧（约1秒）
- **推理速度**: ~30ms (GPU)

## 扩展功能

### 1. 录制视频
可以修改代码添加视频录制功能：

```python
# 在run()方法中添加
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# 在循环中写入帧
out.write(frame)

# 退出时释放
out.release()
```

### 2. 保存预测结果
可以将预测结果保存到文件：

```python
# 添加结果记录
results_log = []

# 在预测后记录
results_log.append({
    'timestamp': time.time(),
    'prediction': prediction,
    'confidence': confidence
})

# 退出时保存
import json
with open('results.json', 'w') as f:
    json.dump(results_log, f)
```

### 3. 多语言支持
可以训练或加载其他语言的模型：
- 中文唇语模型
- 多语言混合模型

## 相关文件

- `realtime_lipreading.py` - 实时检测主程序
- `test_resnet18_dctcn.py` - 离线测试脚本
- `preprocessing/convert_mp4_to_npz.py` - 视频转换工具

## 参考资料

- [原始论文](https://arxiv.org/abs/2001.08702)
- [MediaPipe文档](https://google.github.io/mediapipe/)
- [OpenCV文档](https://docs.opencv.org/)

---

**注意**: 本系统仅供研究和演示使用，实际应用中可能需要针对特定场景进行优化和调整。
