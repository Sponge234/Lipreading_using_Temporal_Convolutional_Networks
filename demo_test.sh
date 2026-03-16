#!/bin/bash
# ResNet18-DCTCN 模型测试示例脚本

echo "=========================================="
echo "ResNet18-DCTCN 模型测试示例"
echo "=========================================="

# 设置路径
PROJECT_ROOT="/root/autodl-tmp/Lipreading_using_Temporal_Convolutional_Networks"
CONFIG_PATH="$PROJECT_ROOT/configs/lrw_resnet18_dctcn.json"
MODEL_PATH="$PROJECT_ROOT/models/lrw_resnet18_dctcn_video.pth"
TEST_DATA_DIR="$PROJECT_ROOT/test_data/processed"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请先下载预训练模型"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

echo ""
echo "步骤1: 检查测试数据..."
if [ -d "$TEST_DATA_DIR" ]; then
    # 统计NPZ文件数量
    npz_count=$(find "$TEST_DATA_DIR" -name "*.npz" | wc -l)
    echo "找到 $npz_count 个测试视频"
    
    if [ $npz_count -eq 0 ]; then
        echo "警告: 没有找到测试数据"
        echo "请参考 test_data/README.md 准备测试数据"
    fi
else
    echo "警告: 测试数据目录不存在"
    echo "请参考 test_data/README.md 准备测试数据"
fi

echo ""
echo "步骤2: 运行测试..."
echo ""

# 示例1: 测试单个视频（如果存在）
sample_video=$(find "$TEST_DATA_DIR" -name "*.npz" | head -n 1)
if [ -n "$sample_video" ]; then
    echo "示例1: 测试单个视频"
    echo "命令: python test_resnet18_dctcn.py --video-path $sample_video --verbose"
    echo ""
    python "$PROJECT_ROOT/test_resnet18_dctcn.py" \
        --config-path "$CONFIG_PATH" \
        --model-path "$MODEL_PATH" \
        --video-path "$sample_video" \
        --verbose
    echo ""
fi

# 示例2: 批量测试（如果有数据）
if [ $npz_count -gt 1 ]; then
    echo ""
    echo "示例2: 批量测试"
    echo "命令: python test_resnet18_dctcn.py --data-dir $TEST_DATA_DIR --split test"
    echo ""
    python "$PROJECT_ROOT/test_resnet18_dctcn.py" \
        --config-path "$CONFIG_PATH" \
        --model-path "$MODEL_PATH" \
        --data-dir "$TEST_DATA_DIR" \
        --split test \
        --output "$PROJECT_ROOT/test_results.json"
    echo ""
fi

echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="
echo ""
echo "更多使用方法请参考: test_data/README.md"
