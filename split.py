"""
数据集划分工具 - 支持随机打乱
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple


class DataSplitter:
    """数据划分器 - 支持完全随机划分"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.text_dir = self.data_dir / "text"
        
    def get_all_instruction_ids(self) -> List[str]:
        """获取所有指令ID"""
        instruction_ids = []
        if self.text_dir.exists():
            for txt_file in sorted(self.text_dir.glob("T*.txt")):
                instruction_ids.append(txt_file.stem)
        return instruction_ids
    
    def split_data(self, train_ratio: float = 0.8, 
                   shuffle: bool = True, seed: int = 42) -> Tuple[List[str], List[str]]:
        """
        划分数据集 - 完全随机
        
        Args:
            train_ratio: 训练集比例
            shuffle: 是否随机打乱
            seed: 随机种子
            
        Returns:
            train_ids, test_ids
        """
        random.seed(seed)
        
        all_ids = self.get_all_instruction_ids()
        total_count = len(all_ids)
        
        print(f"总数据量: {total_count}")
        print(f"训练集比例: {train_ratio*100:.0f}%")
        print(f"随机打乱: {'是' if shuffle else '否'}")
        
        # ⭐ 完全随机打乱
        if shuffle:
            random.shuffle(all_ids)
            print("✓ 数据已随机打乱")
        
        # 计算划分点
        train_size = int(total_count * train_ratio)
        
        train_ids = all_ids[:train_size]
        test_ids = all_ids[train_size:]
        
        print(f"\n最终划分:")
        print(f"  训练集: {len(train_ids)} 条 ({len(train_ids)/total_count*100:.1f}%)")
        print(f"  测试集: {len(test_ids)} 条 ({len(test_ids)/total_count*100:.1f}%)")
        
        return train_ids, test_ids
    
    def save_split(self, train_ids: List[str], test_ids: List[str], 
                    output_dir: str = "."):
        """保存划分结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存训练集ID
        train_file = output_dir / "train_ids.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_ids))
        
        # 保存测试集ID
        test_file = output_dir / "test_ids.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_ids))
        
        # 保存配置
        config = {
            'total_count': len(train_ids) + len(test_ids),
            'train_count': len(train_ids),
            'test_count': len(test_ids),
            'train_ratio': len(train_ids) / (len(train_ids) + len(test_ids)),
            'train_ids_sample': train_ids[:10],
            'test_ids_sample': test_ids[:10]
        }
        
        config_file = output_dir / "split_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"\n划分结果已保存:")
        print(f"  训练集ID: {train_file}")
        print(f"  测试集ID: {test_file}")
        print(f"  配置信息: {config_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='/root/autodl-tmp/data')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./data_split')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LUMI 数据集划分工具（优化版）")
    print("=" * 60)
    
    splitter = DataSplitter(args.data_dir)
    train_ids, test_ids = splitter.split_data(
        args.train_ratio, args.shuffle, args.seed
    )
    splitter.save_split(train_ids, test_ids, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 数据集划分完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()