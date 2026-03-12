"""
LUMI LoRA微调训练脚本 - 优化版
"""

import torch
import os
import json
import argparse
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoFeatureExtractor,
    Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torchaudio
from typing import List, Dict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lumi.constants import DEFAULT_AUDIO_TOKEN
from lumi.model.language_model.lumi_qwen2 import LUMIQwen2Config, LUMIQwen2ForCausalLM
from lumi.util.mm_utils import tokenizer_image_audio_token


class MultiModalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(self, data_dir: str, instruction_ids: List[str],
                 tokenizer, feature_extractor, model_config, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.instruction_ids = instruction_ids
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.model_config = model_config
        self.max_length = max_length
        
        self.audio_dir = self.data_dir / "audio"
        self.text_dir = self.data_dir / "text"
        self.json_dir = self.data_dir / "json"
        
        print(f"加载数据集: {len(self.instruction_ids)} 条指令")
    
    def __len__(self) -> int:
        return len(self.instruction_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inst_id = self.instruction_ids[idx]
        
        try:
            # 加载文本
            text_file = self.text_dir / f"{inst_id}.txt"
            with open(text_file, 'r', encoding='utf-8') as f:
                text_inst = f.read().strip()
            
            # 加载音频
            audio_file = self.audio_dir / f"{inst_id}.wav"
            audio, original_sr = torchaudio.load(str(audio_file))
            if original_sr != 16000:
                resampler = torchaudio.transforms.Resample(original_sr, 16000)
                audio = resampler(audio)
            audio_features = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors='pt'
            )['input_features'].squeeze(0)
            
            # 加载JSON
            json_file = self.json_dir / f"{inst_id}.json"
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 构造提示词
            prompt = f"{DEFAULT_AUDIO_TOKEN}\n用户指令: {text_inst}\n请根据指令生成对应的JSON工作流配置。"
            target_json = json.dumps(json_data, ensure_ascii=False, indent=2)
            full_text = f"{prompt}\n助手回答: {target_json}"
            
            # Token化
            input_ids = tokenizer_image_audio_token(
                full_text, self.tokenizer,
                image_token_index=self.model_config.image_token_index,
                audio_token_index=self.model_config.audio_token_index,
            )
            
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            
            attention_mask = [1] * len(input_ids)
            
            # 创建labels（只计算assistant回答部分的loss）
            prompt_ids = tokenizer_image_audio_token(
                prompt, self.tokenizer,
                image_token_index=self.model_config.image_token_index,
                audio_token_index=self.model_config.audio_token_index,
            )
            prompt_length = len(prompt_ids)
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            
            if len(labels) > self.max_length:
                labels = labels[:self.max_length]
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'audio_features': audio_features,
            }
        except Exception as e:
            print(f"错误: 处理 {inst_id} 时出错: {e}")
            return {
                'input_ids': torch.zeros(10, dtype=torch.long),
                'attention_mask': torch.zeros(10, dtype=torch.long),
                'labels': torch.full((10,), -100, dtype=torch.long),
                'audio_features': torch.zeros(1, 128),
            }


class DataCollator:
    """数据整理器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_length = max(len(f['input_ids']) for f in features)
        
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'audio_features': []
        }
        
        for feature in features:
            padding_length = max_length - len(feature['input_ids'])
            
            batch['input_ids'].append(torch.cat([
                feature['input_ids'],
                torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ]))
            
            batch['attention_mask'].append(torch.cat([
                feature['attention_mask'],
                torch.zeros(padding_length, dtype=torch.long)
            ]))
            
            batch['labels'].append(torch.cat([
                feature['labels'],
                torch.full((padding_length,), -100, dtype=torch.long)
            ]))
            
            batch['audio_features'].append(feature['audio_features'])
        
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'labels': torch.stack(batch['labels']),
            'audio_features': batch['audio_features']
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/data')
    parser.add_argument('--split_dir', type=str, default='./data_split')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/lora_output')
    
    #优化后的参数
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    args = parser.parse_args()
    
    #加载数据划分
    split_dir = Path(args.split_dir)
    with open(split_dir / "train_ids.txt", 'r') as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open(split_dir / "test_ids.txt", 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    
    print(f"训练集: {len(train_ids)} 条")
    print(f"测试集: {len(test_ids)} 条")
    
    #加载tokenizer和特征提取器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_path, subfolder="feature_extractor", trust_remote_code=True
    )
    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    #加载模型
    config_path = os.path.join(args.model_path, 'origin_config.json')
    config = LUMIQwen2Config.from_pretrained(config_path)
    model = LUMIQwen2ForCausalLM.from_pretrained(
        args.model_path, config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    #配置LoRA（优化参数）
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    #准备数据集
    train_dataset = MultiModalDataset(
        args.data_dir, train_ids, tokenizer, 
        feature_extractor, model_config
    )
    eval_dataset = MultiModalDataset(
        args.data_dir, test_ids, tokenizer,
        feature_extractor, model_config
    )
    
    #训练参数（优化）
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        save_total_limit=5,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_torch",
        warmup_steps=200,
        lr_scheduler_type="linear",
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    #创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    
    #开始训练
    print("开始训练...")
    trainer.train()
    
    #保存模型
    final_output_dir = os.path.join(args.output_dir, "final_weights")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"训练完成,权重已保存到: {final_output_dir}")

if __name__ == '__main__':

    main()
