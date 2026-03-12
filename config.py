"""
LUMI LoRA微调配置文件 - 优化版
针对loss下降过快问题进行参数调整
"""

# 路径配置
PATHS = {
    'base_model': '/root/autodl-tmp/LUMI/LUMI_ckpt',
    'data_dir': '/root/autodl-tmp/data',
    'output_dir': '/root/autodl-tmp/lora_output',
    'split_dir': './data_split',
    'eval_dir': './eval_results'
}

# 数据划分配置 
DATA_SPLIT = {
    # 训练集比例
    'train_ratio': 0.8,  # 改为8:2以获得更多训练数据
    'shuffle': True,  # 启用随机打乱，避免简单样本在前
    'random_seed': 42
}

# LoRA训练配置 
LORA_CONFIG = {
    'r': 4, 
    'alpha': 8,  
    'dropout': 0.1, 
    
    'target_modules': [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}

# 训练超参数
TRAINING = {
    'num_epochs': 5, 
    'batch_size': 4,
    'gradient_accumulation_steps': 8,  # 梯度累积
    'learning_rate': 5e-5, 
    'max_length': 512,
    'warmup_steps': 200,  
    'lr_scheduler_type': 'linear',  # 线性衰减  
    'logging_steps': 10,
    'save_steps': 100,
    'eval_steps': 50,  
    'save_total_limit': 5,
    'weight_decay': 0.01,  # 增加L2正则化
}

#  评估配置 
EVALUATION = {
    'max_eval_samples': 200, 
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    'ignore_task_id': True,  # 忽略task_id字段
}

# 可视化配置
VISUALIZATION = {
    'dpi': 300,
    'figsize': (12, 8),
    'generate_excel': True,
    'generate_csv': True,
    'separate_charts': True, 
}


def get_config():
    """获取完整配置字典"""
    return {
        'paths': PATHS,
        'data_split': DATA_SPLIT,
        'lora': LORA_CONFIG,
        'training': TRAINING,
        'evaluation': EVALUATION,
        'visualization': VISUALIZATION
    }


def print_config():
    print("当前配置")
    
    config = get_config()
    for category, settings in config.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':

    print_config()
