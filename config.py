"""
LUMI LoRA微调配置文件 - 优化版
针对loss下降过快问题进行参数调整
"""

# ==================== 路径配置 ====================
PATHS = {
    'base_model': '/root/autodl-tmp/LUMI/LUMI_ckpt',
    'data_dir': '/root/autodl-tmp/data',
    'output_dir': '/root/autodl-tmp/lora_output',
    'split_dir': './data_split',
    'eval_dir': './eval_results'
}

# ==================== 数据划分配置 ====================
DATA_SPLIT = {
    # 训练集比例
    'train_ratio': 0.8,  # 改为8:2以获得更多训练数据
    
    # 是否随机打乱数据（⭐重要修改）
    'shuffle': True,  # 启用随机打乱，避免简单样本在前
    
    # 随机种子
    'random_seed': 42
}

# ==================== LoRA训练配置 ====================
LORA_CONFIG = {
    # ⭐ 降低LoRA秩以减少过拟合
    'r': 4,  # 从8降到4，减少可训练参数
    
    # ⭐ 调整alpha
    'alpha': 8,  # 从16降到8，与r保持2倍关系
    
    # ⭐ 增加dropout
    'dropout': 0.1,  # 从0.05增加到0.1，增强正则化
    
    'target_modules': [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}

# ==================== 训练超参数 ====================
TRAINING = {
    # ⭐ 增加训练轮数
    'num_epochs': 5,  # 从3增加到5，让模型更充分学习
    
    'batch_size': 4,
    
    # ⭐ 增加梯度累积
    'gradient_accumulation_steps': 8,  # 从4增加到8
    
    # ⭐⭐⭐ 降低学习率（最重要的修改）
    'learning_rate': 5e-5,  # 从2e-4降到5e-5，大幅降低
    
    'max_length': 512,
    
    # ⭐ 增加warmup
    'warmup_steps': 200,  # 从100增加到200
    
    # ⭐ 使用线性衰减而非cosine
    'lr_scheduler_type': 'linear',  # 改为linear，更平稳
    
    'logging_steps': 10,
    'save_steps': 100,
    'eval_steps': 50,  # 更频繁评估
    'save_total_limit': 5,
    
    # ⭐ 添加权重衰减
    'weight_decay': 0.01,  # 增加L2正则化
}

# ==================== 评估配置 ====================
EVALUATION = {
    # ⭐ 增加评估样本数
    'max_eval_samples': 200,  # 从100增加到200
    
    'max_new_tokens': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    
    # ⭐ 添加任务ID忽略配置
    'ignore_task_id': True,  # 忽略task_id字段
}

# ==================== 可视化配置 ====================
VISUALIZATION = {
    'dpi': 300,
    'figsize': (12, 8),
    'generate_excel': True,
    'generate_csv': True,
    
    # ⭐ 添加单独图表配置
    'separate_charts': True,  # 生成独立图表
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
    """打印当前配置"""
    print("=" * 60)
    print("当前配置（优化版）")
    print("=" * 60)
    
    config = get_config()
    for category, settings in config.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    print_config()