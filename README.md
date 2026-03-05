# LUMI LoRA Multimodal Fine-tuning (Audio + Text → JSON)

本仓库提供一套用于 **LUMI (Qwen2-based) 多模态模型** 的 LoRA 微调与评估脚本，面向“中文语音/文本指令 → JSON 工作流配置”任务。

包含：
- 数据集随机划分（train/test）
- LoRA 微调训练（Trainer）
- 测试集评估（支持忽略 task_id）
- 生成可视化与导出表格（PNG / CSV / Excel）
- 从 Trainer 日志绘制 loss 曲线

---

## 1. 项目结构

建议仓库结构如下：
.
├── config.py # 训练/评估/可视化参数集中配置（可选使用）
├── split.py # 随机划分训练/测试集（输出 train_ids.txt/test_ids.txt）
├── train.py # LoRA 微调训练脚本
├── eval.py # 测试集评估 + 可视化输出
├── run.sh # 一键跑完整流程：split → train → eval → report
├── data/ 
│ ├── audio/ 
│ ├── text/ 
│ └── json/ 
├── data_split/ # split.py 输出：train_ids.txt / test_ids.txt / split_config.json
├── lora_output/ # train.py 输出：checkpoint / final_weights
└── eval_results/ # eval.py 输出：eval_results.json + 图表 + 表格


---

## 2. 数据集格式要求（关键）

脚本按 **ID 对齐三模态文件**：

- `data/text/T00001.txt`：指令文本（中文）
- `data/audio/T00001.wav`：语音（会被重采样到 16kHz）
- `data/json/T00001.json`：目标 JSON（训练时作为“助手回答”监督信号）

ID 规则：`Txxxxx`（例如 T00001、T09778）。

---

## 3. 环境依赖

### Python 包（核心）
- torch
- transformers
- peft
- torchaudio
- numpy
- pandas
- matplotlib
- tqdm
- openpyxl（用于导出 xlsx）

建议使用 Python 3.10+。

### 字体（可选）
评估脚本尝试使用 `SimHei` 显示中文；没有也可以运行，只是中文可能显示为方块。

---

## 4. 快速开始（推荐：一键脚本）

### 4.1 修改 run.sh 里的路径
打开 `run.sh`，确认：
- `BASE_MODEL` 指向你的 LUMI 基座模型目录（包含 origin_config.json 等）
- `DATA_DIR` 指向你的数据目录（含 audio/text/json）

### 4.2 执行
```bash
bash run.sh

run.sh 默认流程：

数据划分（随机打乱，80/20）

LoRA 训练（默认 5 epoch、lr=5e-5、r=4、alpha=8、weight_decay=0.01）

测试集评估（忽略 task_id，输出 accuracy/precision/recall/f1）

生成训练报告 training_report.txt

5. 分步执行
5.1 数据划分
python split.py \
  --data_dir /path/to/data \
  --train_ratio 0.8 \
  --shuffle \
  --seed 42 \
  --output_dir ./data_split

输出：

data_split/train_ids.txt

data_split/test_ids.txt

data_split/split_config.json

5.2 训练
python train.py \
  --model_path /path/to/base_model \
  --data_dir /path/to/data \
  --split_dir ./data_split \
  --output_dir ./lora_output \
  --num_epochs 5 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --lora_r 4 \
  --lora_alpha 8 \
  --weight_decay 0.01

训练输出（默认）：

lora_output/final_weights/（LoRA 权重 + tokenizer）

5.3 评估 + 可视化
python eval.py \
  --base_model /path/to/base_model \
  --lora_weights ./lora_output/final_weights \
  --data_dir /path/to/data \
  --split_dir ./data_split \
  --output_dir ./eval_results

评估会：

对每条样本推理生成 JSON

解析 {...} 并与 reference 对比

默认忽略 task_id 字段（更适合你的数据生成逻辑）

输出总体指标 accuracy/precision/recall/f1

生成图表与表格