#!/bin/bash
# 实时唇语检测启动脚本

echo "=========================================="
echo "实时摄像头唇语检测系统"
echo "=========================================="
echo ""

# 检查依赖
echo "检查依赖..."
python -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: OpenCV未安装"
    echo "请运行: pip install opencv-python"
    exit 1
fi

python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: PyTorch未安装"
    echo "请运行: pip install torch"
    exit 1
fi

python -c "import mediapipe" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: MediaPipe未安装，将使用OpenCV检测器"
    echo "建议安装: pip install mediapipe"
    USE_MEDIAPIPE="--use-mediapipe False"
else
    echo "✓ MediaPipe已安装"
    USE_MEDIAPIPE=""
fi

# 检查模型文件
if [ ! -f "./models/lrw_resnet18_dctcn_video.pth" ]; then
    echo "错误: 模型文件不存在"
    echo "请确保模型文件位于: ./models/lrw_resnet18_dctcn_video.pth"
    exit 1
fi

echo "✓ 所有依赖已就绪"
echo ""

# 检查摄像头
echo "检查摄像头..."
if [ -e "/dev/video0" ]; then
    echo "✓ 检测到摄像头 /dev/video0"
    CAMERA_ID=0
elif [ -e "/dev/video1" ]; then
    echo "✓ 检测到摄像头 /dev/video1"
    CAMERA_ID=1
else
    echo "警告: 未检测到摄像头设备"
    echo "将尝试使用默认摄像头ID (0)"
    CAMERA_ID=0
fi

echo ""
echo "启动实时检测..."
echo "控制说明:"
echo "  q - 退出"
echo "  r - 重置缓冲"
echo "  s - 保存当前帧"
echo ""
echo "=========================================="
echo ""

# 运行实时检测
python realtime_lipreading.py \
    --camera-id $CAMERA_ID \
    --buffer-size 29 \
    $USE_MEDIAPIPE

echo ""
echo "=========================================="
echo "检测已结束"
echo "=========================================="
