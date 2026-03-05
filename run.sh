#!/bin/bash
# LUMI LoRA微调完整流程 - 优化版

echo "=============================================="
echo "LUMI LoRA微调完整流程（优化版）"
echo "=============================================="

# 配置参数
BASE_MODEL="/root/autodl-tmp/LUMI/LUMI_ckpt"
DATA_DIR="/root/autodl-tmp/data"
OUTPUT_DIR="/root/autodl-tmp/lora_output"
SPLIT_DIR="./data_split"
EVAL_DIR="./eval_results"

# ⭐ 优化后的训练参数
TRAIN_RATIO=0.8
NUM_EPOCHS=5
BATCH_SIZE=4
LEARNING_RATE=5e-5
LORA_R=4
LORA_ALPHA=8
WEIGHT_DECAY=0.01

echo ""
echo "==================== 配置信息 ===================="
echo "基础模型: $BASE_MODEL"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "训练比例: 80% (训练) : 20% (测试)"
echo "训练轮数: $NUM_EPOCHS"
echo "学习率: $LEARNING_RATE (⭐优化降低)"
echo "LoRA秩: $LORA_R (⭐优化降低)"
echo "权重衰减: $WEIGHT_DECAY (⭐新增)"
echo "================================================="

# 步骤1: 数据划分（随机）
echo ""
echo "[步骤 1/4] 数据集随机划分..."
echo "================================================="

python split.py \
    --data_dir $DATA_DIR \
    --train_ratio $TRAIN_RATIO \
    --shuffle \
    --seed 42 \
    --output_dir $SPLIT_DIR

if [ $? -ne 0 ]; then
    echo "❌ 错误: 数据划分失败"
    exit 1
fi

echo "✅ 数据划分完成"

# 步骤2: LoRA训练
echo ""
echo "[步骤 2/4] 开始LoRA微调训练..."
echo "================================================="

python train.py \
    --model_path $BASE_MODEL \
    --data_dir $DATA_DIR \
    --split_dir $SPLIT_DIR \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --weight_decay $WEIGHT_DECAY

if [ $? -ne 0 ]; then
    echo "❌ 错误: 训练失败"
    exit 1
fi

echo "✅ 训练完成"

# 步骤3: 模型评估
echo ""
echo "[步骤 3/4] 在测试集上评估模型..."
echo "================================================="

LORA_WEIGHTS="$OUTPUT_DIR/final_weights"

if [ -d "$LORA_WEIGHTS" ]; then
    python eval.py \
        --base_model $BASE_MODEL \
        --lora_weights $LORA_WEIGHTS \
        --data_dir $DATA_DIR \
        --split_dir $SPLIT_DIR \
        --output_dir $EVAL_DIR
    
    if [ $? -eq 0 ]; then
        echo "✅ 评估完成"
    else
        echo "⚠️  警告: 评估失败"
    fi
else
    echo "⚠️  警告: 找不到LoRA权重,跳过评估"
fi

# 步骤4: 生成报告
echo ""
echo "[步骤 4/4] 生成训练报告..."
echo "================================================="

REPORT_FILE="$OUTPUT_DIR/training_report.txt"

cat > $REPORT_FILE << EOF
================================================
LUMI LoRA 微调训练报告（优化版）
================================================
生成时间: $(date)

优化措施:
  ⭐ 学习率降低: 2e-4 -> 5e-5
  ⭐ LoRA秩降低: 8 -> 4
  ⭐ 增加Dropout: 0.05 -> 0.1
  ⭐ 增加权重衰减: 0 -> 0.01
  ⭐ 数据随机打乱
  ⭐ 增加训练轮数: 3 -> 5

配置信息:
  - 基础模型: $BASE_MODEL
  - 数据目录: $DATA_DIR
  - 训练比例: 80% : 20%
  - 学习率: $LEARNING_RATE
  - LoRA秩: $LORA_R
  - 权重衰减: $WEIGHT_DECAY

输出文件:
  - LoRA权重: $LORA_WEIGHTS
  - 评估结果: $EVAL_DIR

EOF

# 添加评估指标
if [ -f "$EVAL_DIR/eval_results.json" ]; then
    python3 << PYEOF >> $REPORT_FILE
import json
with open("$EVAL_DIR/eval_results.json", "r") as f:
    results = json.load(f)
    overall = results.get("overall", {})
    print("评估指标:")
    print(f"  准确率: {overall.get('accuracy', 0):.2f}%")
    print(f"  精确率: {overall.get('precision', 0):.2f}%")
    print(f"  召回率: {overall.get('recall', 0):.2f}%")
    print(f"  F1分数: {overall.get('f1', 0):.2f}%")
    print(f"  测试样本数: {overall.get('total_samples', 0)}")
PYEOF
fi

cat >> $REPORT_FILE << EOF

================================================
训练完成时间: $(date)
================================================
EOF

echo "✅ 报告已生成: $REPORT_FILE"

# 最终总结
echo ""
echo "================================================="
echo "🎉 完整流程执行完成!"
echo "================================================="
echo ""
echo "输出文件位置:"
echo "  📁 LoRA权重: $LORA_WEIGHTS"
echo "  📊 准确率图表: $EVAL_DIR/accuracy_chart.png"
echo "  📊 精确率图表: $EVAL_DIR/precision_chart.png"
echo "  📊 F1分数图表: $EVAL_DIR/f1_chart.png"
echo "  📊 综合对比图: $EVAL_DIR/overall_metrics.png"
echo "  📋 详细表格: $EVAL_DIR/detailed_results.csv"
echo "  📄 训练报告: $REPORT_FILE"
echo ""
echo "================================================="
echo "下一步操作:"
echo "  1. 查看loss曲线确认是否改善"
echo "  2. 检查可视化图表分析性能"
echo "  3. 根据需要继续调整参数"
echo "================================================="