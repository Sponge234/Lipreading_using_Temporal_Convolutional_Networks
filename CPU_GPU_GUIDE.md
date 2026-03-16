# CPU/GPU 设备切换使用指南

## 概述

本项目已完全支持CPU和GPU的自动切换，可以在没有GPU的环境下正常运行。系统会自动检测可用设备，也可以手动指定使用CPU或GPU。

## 设备选择模式

### 1. 自动模式（推荐）
系统自动检测并选择最优设备：
- 如果检测到CUDA可用，自动使用GPU
- 如果GPU不可用，自动切换到CPU

```bash
# 自动选择设备（默认）
python realtime_lipreading.py --device auto
python test_resnet18_dctcn.py --device auto
```

### 2. 强制GPU模式
强制使用GPU，如果GPU不可用会自动降级到CPU：

```bash
python realtime_lipreading.py --device cuda
python test_resnet18_dctcn.py --device cuda
```

### 3. 强制CPU模式
强制使用CPU，即使GPU可用：

```bash
python realtime_lipreading.py --device cpu
python test_resnet18_dctcn.py --device cpu
```

## CPU优化配置

### 1. 多线程优化
代码已自动配置CPU多线程优化：
```python
# 自动设置4个CPU线程
torch.set_num_threads(4)
```

### 2. 手动调整线程数
根据CPU核心数调整：

```python
# 查看CPU核心数
import os
print(f"CPU核心数: {os.cpu_count()}")

# 建议设置
# 4核CPU: torch.set_num_threads(4)
# 8核CPU: torch.set_num_threads(6)
# 16核CPU: torch.set_num_threads(8)
```

### 3. 内存优化
CPU模式下内存使用优化：

```python
# 减小batch size
--batch-size 1

# 减小缓冲大小
--buffer-size 20
```

## 性能对比

### GPU模式（推荐）
- **实时检测FPS**: 25-30
- **单次推理时间**: ~30ms
- **批量测试速度**: 快
- **内存占用**: GPU显存 ~2GB

### CPU模式
- **实时检测FPS**: 5-10
- **单次推理时间**: ~100-200ms
- **批量测试速度**: 较慢
- **内存占用**: 系统内存 ~4GB

## 使用建议

### 有GPU的情况
```bash
# 使用默认自动模式即可
python realtime_lipreading.py

# 或明确指定GPU
python realtime_lipreading.py --device cuda
```

### 无GPU的情况
```bash
# 自动使用CPU
python realtime_lipreading.py

# 或明确指定CPU
python realtime_lipreading.py --device cpu

# CPU模式下建议降低帧率期望
# FPS会在5-10左右，这是正常的
```

### CPU性能优化建议

1. **减少缓冲大小**
```bash
# 从默认29帧减少到20帧
python realtime_lipreading.py --buffer-size 20
```

2. **降低视频分辨率**
```python
# 修改代码中的摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 从640降到320
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 从480降到240
```

3. **使用OpenCV检测器**
```bash
# OpenCV检测器比MediaPipe更快
python realtime_lipreading.py --use-mediapipe False
```

4. **关闭显示窗口**
```bash
# 不显示实时画面，只输出结果
python realtime_lipreading.py --no-display
```

## 检查设备状态

### 检查CUDA是否可用
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
```

### 检查CPU信息
```python
import os
import multiprocessing
print(f"CPU核心数: {os.cpu_count()}")
print(f"当前线程数: {torch.get_num_threads()}")
```

## 常见问题

### Q1: 如何知道当前使用的是CPU还是GPU？

**A**: 程序启动时会显示：
```
✓ 使用GPU: NVIDIA GeForce RTX 3090
# 或
✓ 使用CPU (GPU不可用)
```

### Q2: CPU模式下FPS很低怎么办？

**A**: 这是正常的，CPU模式性能有限。可以：
- 降低buffer-size
- 使用OpenCV检测器
- 降低视频分辨率
- 接受较低的FPS（5-10）

### Q3: 如何强制使用CPU？

**A**: 使用 `--device cpu` 参数：
```bash
python realtime_lipreading.py --device cpu
```

### Q4: CPU模式内存不足怎么办？

**A**: 
- 减小batch-size
- 减小buffer-size
- 关闭其他程序
- 增加系统swap空间

### Q5: 多核CPU如何优化？

**A**: 代码已自动设置4线程，可以手动调整：
```python
# 在代码中修改
torch.set_num_threads(8)  # 根据CPU核心数调整
```

## 性能测试

### 测试脚本
```bash
# 测试GPU性能
python test_resnet18_dctcn.py --device cuda --video-path test_data/processed/ABOUT/test/sample_1.npz

# 测试CPU性能
python test_resnet18_dctcn.py --device cpu --video-path test_data/processed/ABOUT/test/sample_1.npz
```

### 预期结果
- **GPU**: 推理时间 ~30ms
- **CPU**: 推理时间 ~100-200ms

## 设备切换示例

### 实时检测
```bash
# 自动选择（推荐）
python realtime_lipreading.py

# 强制GPU
python realtime_lipreading.py --device cuda

# 强制CPU
python realtime_lipreading.py --device cpu
```

### 离线测试
```bash
# 自动选择
python test_resnet18_dctcn.py --video-path test.npz

# 强制GPU
python test_resnet18_dctcn.py --device cuda --video-path test.npz

# 强制CPU
python test_resnet18_dctcn.py --device cpu --video-path test.npz
```

## 技术细节

### 设备检测逻辑
```python
def setup_device(device='auto'):
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("GPU不可用，切换到CPU")
            return torch.device('cpu')
    else:
        return torch.device('cpu')
```

### 数据传输
```python
# GPU模式
input_tensor = data.cuda()

# CPU模式
input_tensor = data.to(device)  # 自动适配

# 通用写法（推荐）
input_tensor = data.to(self.device)
```

## 总结

- ✅ 支持CPU/GPU自动切换
- ✅ CPU模式已优化（多线程）
- ✅ 可手动指定设备
- ✅ GPU不可用时自动降级
- ✅ 性能差异：GPU快3-10倍

**推荐**: 有GPU时使用GPU，无GPU时CPU也能正常运行（性能会降低）。

---

**更新时间**: 2026-03-17
**支持版本**: 所有脚本均已支持CPU/GPU切换
