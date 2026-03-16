# 快速开始指南

##  安装依赖
```bash
pip install -r requirements.txt
pip install scikit-image
pip install "mediapipe<=0.10.8"
```

## 1. 测试环境检查

首先检查模型和配置文件是否存在：

```bash
# 检查模型文件
ls -lh models/lrw_resnet18_dctcn_video.pth

# 检查配置文件
cat configs/lrw_resnet18_dctcn.json
```

## 2. 准备测试数据

### 方式A: 使用现有NPZ数据（如果有）

如果你已经有处理好的NPZ文件：

```bash
# 创建目录结构
mkdir -p test_data/processed/WORD/test/

# 复制NPZ文件
cp /path/to/your/data.npz test_data/processed/WORD/test/
```

### 方式B: 从MP4视频转换

如果你有原始MP4视频：

```bash
# 1. 复制视频到测试目录
cp /path/to/your/videos/*.mp4 test_data/raw_videos/

# 2. 转换为NPZ格式（使用MediaPipe自动检测人脸）
python preprocessing/convert_mp4_to_npz.py \
    --video-direc test_data/raw_videos \
    --output-direc test_data/processed \
    --auto-detect \
    --detector mediapipe
```

## 3. 运行测试

### 测试单个视频

```bash
python test_resnet18_dctcn.py \
    --video-path test_data/processed/ABOUT/test/video1.npz \
    --verbose
```

### 批量测试

```bash
python test_resnet18_dctcn.py \
    --data-dir test_data/processed \
    --split test \
    --output results.json
```

## 4. 查看结果

测试结果会显示：
- 预测的单词类别
- 置信度分数
- Top-5 预测结果
- 整体准确率（批量测试）

## 常用命令

```bash
# 查看帮助
python test_resnet18_dctcn.py --help

# 使用示例脚本
./demo_test.sh

# 查看详细文档
cat test_data/README.md
```

## 下一步

- 准备更多测试数据以提高测试可靠性
- 尝试不同的视频质量，观察模型性能
- 分析错误案例，了解模型的局限性
