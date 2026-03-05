"""
评估和可视化脚本 - 优化版
支持忽略task_id、生成独立图表
"""

import torch
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig, AutoFeatureExtractor
import torchaudio
from tqdm import tqdm
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lumi.constants import DEFAULT_AUDIO_TOKEN
from lumi.model.language_model.lumi_qwen2 import LUMIQwen2Config, LUMIQwen2ForCausalLM
from lumi.util.mm_utils import tokenizer_image_audio_token

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_lora_model(base_model_path: str, lora_weights_path: str):
    """加载LoRA模型"""
    print("=" * 60)
    print("加载LoRA模型...")
    print("=" * 60)
    
    config_path = os.path.join(base_model_path, 'origin_config.json')
    config = LUMIQwen2Config.from_pretrained(config_path)
    
    base_model = LUMIQwen2ForCausalLM.from_pretrained(
        base_model_path, config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(
        base_model, lora_weights_path,
        torch_dtype=torch.bfloat16
    )
    
    model = model.merge_and_unload()
    print("✓ 模型加载完成")
    return model


def process_audio(audio_path: str, feature_extractor):
    """处理音频"""
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    return feature_extractor(audio, sampling_rate=16000, return_tensors='pt')['input_features']


def inference(model, tokenizer, feature_extractor, model_config,
              text_instruction: str, audio_path: str = None):
    """执行推理"""
    device = next(model.parameters()).device
    
    prompt = ""
    if audio_path:
        prompt += f"{DEFAULT_AUDIO_TOKEN}\n"
    prompt += f"用户指令: {text_instruction}\n请根据指令生成对应的JSON工作流配置。\n助手回答: "
    
    input_ids = tokenizer_image_audio_token(
        prompt, tokenizer,
        image_token_index=model_config.image_token_index,
        audio_token_index=model_config.audio_token_index,
    )
    input_ids = torch.tensor([input_ids]).to(device)
    
    if audio_path:
        audio_features = process_audio(audio_path, feature_extractor).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "助手回答:" in generated_text:
        response = generated_text.split("助手回答:")[-1].strip()
    else:
        response = generated_text
    
    return response


def compare_json_ignore_task_id(pred_dict: dict, ref_dict: dict) -> tuple:
    """
    ⭐ 比较JSON，忽略task_id字段
    返回: (exact_match, precision, recall, f1)
    """
    # 移除task_id
    pred_copy = {k: v for k, v in pred_dict.items() if k != 'task_id'}
    ref_copy = {k: v for k, v in ref_dict.items() if k != 'task_id'}
    
    # 转换为字符串集合进行比较
    def dict_to_items(d, prefix=''):
        items = set()
        for k, v in d.items():
            if isinstance(v, dict):
                items.update(dict_to_items(v, f"{prefix}{k}."))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(dict_to_items(item, f"{prefix}{k}[{i}]."))
                    else:
                        items.add(f"{prefix}{k}[{i}]:{item}")
            else:
                items.add(f"{prefix}{k}:{v}")
        return items
    
    pred_items = dict_to_items(pred_copy)
    ref_items = dict_to_items(ref_copy)
    
    # 精确匹配
    exact_match = (pred_items == ref_items)
    
    # 计算F1
    if not ref_items:
        return exact_match, 0, 0, 0
    
    tp = len(pred_items & ref_items)
    fp = len(pred_items - ref_items)
    fn = len(ref_items - pred_items)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return exact_match, precision, recall, f1


def evaluate_on_test_set(model, tokenizer, feature_extractor, model_config,
                          data_dir: str, test_ids: List[str], output_dir: str):
    """在测试集上评估"""
    print("\n" + "=" * 60)
    print("开始评估...")
    print("=" * 60)
    
    data_dir = Path(data_dir)
    results = []
    
    model.eval()
    
    for inst_id in tqdm(test_ids, desc="评估进度"):
        try:
            # 加载数据
            text_file = data_dir / "text" / f"{inst_id}.txt"
            with open(text_file, 'r', encoding='utf-8') as f:
                text_inst = f.read().strip()
            
            json_file = data_dir / "json" / f"{inst_id}.json"
            with open(json_file, 'r', encoding='utf-8') as f:
                ref_json = json.load(f)
            
            # 推理
            audio_file = data_dir / "audio" / f"{inst_id}.wav"
            audio_path = str(audio_file) if audio_file.exists() else None
            
            pred = inference(model, tokenizer, feature_extractor, 
                           model_config, text_inst, audio_path)
            
            # 提取JSON
            match = re.search(r'\{[\s\S]*\}', pred)
            if match:
                pred_json_str = match.group(0)
                pred_json = json.loads(pred_json_str)
            else:
                pred_json = {}
            
            # ⭐ 比较JSON（忽略task_id）
            exact_match, precision, recall, f1 = compare_json_ignore_task_id(
                pred_json, ref_json
            )
            
            results.append({
                'instruction_id': inst_id,
                'exact_match': exact_match,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'prediction': pred_json_str if match else pred,
                'reference': json.dumps(ref_json, ensure_ascii=False)
            })
            
        except Exception as e:
            print(f"\n错误: {inst_id} - {e}")
            results.append({
                'instruction_id': inst_id,
                'exact_match': False,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'error': str(e)
            })
    
    # 计算总体指标
    accuracy = sum(r['exact_match'] for r in results) / len(results) * 100
    avg_precision = np.mean([r['precision'] for r in results]) * 100
    avg_recall = np.mean([r['recall'] for r in results]) * 100
    avg_f1 = np.mean([r['f1'] for r in results]) * 100
    
    overall = {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'total_samples': len(results)
    }
    
    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "eval_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({'overall': overall, 'per_sample': results}, 
                  f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果:")
    print(f"  准确率: {accuracy:.2f}%")
    print(f"  精确率: {avg_precision:.2f}%")
    print(f"  召回率: {avg_recall:.2f}%")
    print(f"  F1分数: {avg_f1:.2f}%")
    
    return results, overall


def create_visualizations(results: List[Dict], overall: Dict, output_dir: Path):
    """⭐ 创建独立的可视化图表"""
    print("\n生成可视化图表...")
    
    # 提取数据
    sample_indices = list(range(len(results)))
    accuracies = [r['exact_match'] * 100 for r in results]
    precisions = [r['precision'] * 100 for r in results]
    recalls = [r['recall'] * 100 for r in results]
    f1_scores = [r['f1'] * 100 for r in results]
    
    # 1️⃣ 准确率折线图
    plt.figure(figsize=(10, 6))
    plt.plot(sample_indices, accuracies, 'b-', linewidth=2, alpha=0.7)
    plt.axhline(y=overall['accuracy'], color='r', linestyle='--', 
                label=f'平均: {overall["accuracy"]:.2f}%')
    plt.xlabel('样本编号', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title('测试集准确率分布', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_chart.png", dpi=300)
    plt.close()
    
    # 2️⃣ 精确率折线图
    plt.figure(figsize=(10, 6))
    plt.plot(sample_indices, precisions, 'g-', linewidth=2, alpha=0.7)
    plt.axhline(y=overall['precision'], color='r', linestyle='--',
                label=f'平均: {overall["precision"]:.2f}%')
    plt.xlabel('样本编号', fontsize=12)
    plt.ylabel('精确率 (%)', fontsize=12)
    plt.title('测试集精确率分布', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_chart.png", dpi=300)
    plt.close()
    
    # 3️⃣ F1分数折线图
    plt.figure(figsize=(10, 6))
    plt.plot(sample_indices, f1_scores, 'orange', linewidth=2, alpha=0.7)
    plt.axhline(y=overall['f1'], color='r', linestyle='--',
                label=f'平均: {overall["f1"]:.2f}%')
    plt.xlabel('样本编号', fontsize=12)
    plt.ylabel('F1分数 (%)', fontsize=12)
    plt.title('测试集F1分数分布', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "f1_chart.png", dpi=300)
    plt.close()
    
    # 4️⃣ 综合对比柱状图
    plt.figure(figsize=(10, 6))
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    values = [overall['accuracy'], overall['precision'], 
              overall['recall'], overall['f1']]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('分数 (%)', fontsize=12)
    plt.title('总体评估指标对比', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_metrics.png", dpi=300)
    plt.close()
    
    # 5️⃣ 创建详细表格
    df = pd.DataFrame([{
        '指令ID': r['instruction_id'],
        '精确匹配': '是' if r['exact_match'] else '否',
        '精确率(%)': f"{r['precision']*100:.2f}",
        '召回率(%)': f"{r['recall']*100:.2f}",
        'F1分数(%)': f"{r['f1']*100:.2f}",
    } for r in results])
    
    csv_file = output_dir / "detailed_results.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    try:
        excel_file = output_dir / "detailed_results.xlsx"
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✓ Excel表格: {excel_file}")
    except:
        pass
    
    print(f"✓ 准确率图表: {output_dir / 'accuracy_chart.png'}")
    print(f"✓ 精确率图表: {output_dir / 'precision_chart.png'}")
    print(f"✓ F1分数图表: {output_dir / 'f1_chart.png'}")
    print(f"✓ 综合对比图: {output_dir / 'overall_metrics.png'}")
    print(f"✓ CSV表格: {csv_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_weights', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/data')
    parser.add_argument('--split_dir', type=str, default='./data_split')
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_lora_model(args.base_model, args.lora_weights)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.base_model, subfolder="feature_extractor", trust_remote_code=True
    )
    model_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    
    # 加载测试集ID
    with open(Path(args.split_dir) / "test_ids.txt", 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    
    # 评估
    results, overall = evaluate_on_test_set(
        model, tokenizer, feature_extractor, model_config,
        args.data_dir, test_ids, args.output_dir
    )
    
    # 可视化
    create_visualizations(results, overall, Path(args.output_dir))
    
    print("\n" + "=" * 60)
    print("✅ 评估完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()