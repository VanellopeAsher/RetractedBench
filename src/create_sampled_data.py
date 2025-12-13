#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据RetractionSubject字段中的subject比例进行采样的脚本
从matched_papers.csv中采样1000条记录，保持各subject的比例分布
"""

import pandas as pd
import re
from collections import Counter, defaultdict
import random
import numpy as np

def extract_subjects(retraction_subject):
    """
    从RetractionSubject字段中提取subject列表
    格式: (ENV) Environmental Sciences;(ENV) Ground/Surface Water;(PHY) Engineering - Chemical;
    返回: ['ENV', 'ENV', 'PHY'] 这样的列表
    """
    if pd.isna(retraction_subject) or retraction_subject == '':
        return []
    
    # 使用正则表达式提取括号内的内容
    pattern = r'\(([^)]+)\)'
    subjects = re.findall(pattern, str(retraction_subject))
    return subjects

def analyze_subject_distribution(df):
    """
    分析整个数据集中各subject的分布
    """
    subject_counts = Counter()
    subject_records = defaultdict(list)  # 记录每个subject对应的行索引
    
    print("正在分析subject分布...")
    
    for idx, row in df.iterrows():
        subjects = extract_subjects(row['RetractionSubject'])
        for subject in subjects:
            subject_counts[subject] += 1
            subject_records[subject].append(idx)
    
    total_subjects = sum(subject_counts.values())
    subject_proportions = {subject: count/total_subjects for subject, count in subject_counts.items()}
    
    print(f"总共找到 {len(subject_counts)} 个不同的subject")
    print(f"总subject出现次数: {total_subjects}")
    print("\nTop 20 subjects及其比例:")
    for subject, count in subject_counts.most_common(20):
        proportion = subject_proportions[subject]
        print(f"{subject}: {count} ({proportion:.4f})")
    
    return subject_counts, subject_proportions, subject_records

def stratified_sampling(df, target_size=1000, random_seed=42):
    """
    根据subject比例进行分层采样
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\n开始分层采样，目标样本数: {target_size}")
    
    # 分析subject分布
    subject_counts, subject_proportions, subject_records = analyze_subject_distribution(df)
    
    # 计算每个subject应该采样的数量
    subject_sample_sizes = {}
    for subject, proportion in subject_proportions.items():
        target_count = int(proportion * target_size)
        available_count = len(subject_records[subject])
        # 不能超过可用数量
        actual_count = min(target_count, available_count)
        subject_sample_sizes[subject] = actual_count
    
    print(f"\n各subject采样计划:")
    total_planned = 0
    for subject in sorted(subject_sample_sizes.keys()):
        planned = subject_sample_sizes[subject]
        total_planned += planned
        print(f"{subject}: {planned} 条")
    print(f"计划总采样数: {total_planned}")
    
    # 如果计划采样数不足目标数，随机补充
    if total_planned < target_size:
        remaining = target_size - total_planned
        print(f"需要随机补充 {remaining} 条记录")
        
        # 从所有记录中随机选择补充
        all_indices = set(range(len(df)))
        used_indices = set()
        
        # 收集已使用的索引
        for subject, indices in subject_records.items():
            sample_size = subject_sample_sizes[subject]
            if sample_size > 0:
                sampled_indices = random.sample(indices, sample_size)
                used_indices.update(sampled_indices)
        
        # 从未使用的索引中随机选择补充
        available_indices = list(all_indices - used_indices)
        if len(available_indices) >= remaining:
            additional_indices = random.sample(available_indices, remaining)
            used_indices.update(additional_indices)
        else:
            # 如果可用索引不足，从所有索引中随机选择
            additional_indices = random.sample(list(all_indices), remaining)
            used_indices.update(additional_indices)
    
    # 执行采样
    sampled_indices = set()
    
    for subject, indices in subject_records.items():
        sample_size = subject_sample_sizes[subject]
        if sample_size > 0 and len(indices) >= sample_size:
            sampled = random.sample(indices, sample_size)
            sampled_indices.update(sampled)
    
    # 如果还需要补充
    if len(sampled_indices) < target_size:
        remaining = target_size - len(sampled_indices)
        all_indices = set(range(len(df)))
        available = list(all_indices - sampled_indices)
        if len(available) >= remaining:
            additional = random.sample(available, remaining)
            sampled_indices.update(additional)
        else:
            # 如果还是不够，从所有记录中随机选择
            additional = random.sample(list(all_indices), remaining)
            sampled_indices.update(additional)
    
    # 创建采样后的DataFrame
    sampled_df = df.iloc[list(sampled_indices)].copy()
    
    print(f"\n实际采样结果:")
    print(f"采样记录数: {len(sampled_df)}")
    
    # 验证采样后的subject分布
    sampled_subject_counts = Counter()
    for _, row in sampled_df.iterrows():
        subjects = extract_subjects(row['RetractionSubject'])
        for subject in subjects:
            sampled_subject_counts[subject] += 1
    
    print(f"\n采样后的subject分布 (Top 10):")
    for subject, count in sampled_subject_counts.most_common(10):
        proportion = count / sum(sampled_subject_counts.values())
        print(f"{subject}: {count} ({proportion:.4f})")
    
    return sampled_df

def main():
    """
    主函数
    """
    input_file = 'sampled_matched_papers_10k.csv'
    output_file = 'sampled_matched_papers_0.2k.csv'
    target_size = 200
    
    print("=== 根据Subject比例进行分层采样 ===")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"目标采样数: {target_size}")
    
    # 读取数据
    try:
        print(f"\n正在读取 {input_file}...")
        df = pd.read_csv(input_file, low_memory=False)
        print(f"成功读取 {len(df)} 条记录")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 检查RetractionSubject字段
    if 'RetractionSubject' not in df.columns:
        print("错误: 找不到RetractionSubject字段")
        return
    
    # 执行分层采样
    sampled_df = stratified_sampling(df, target_size)
    
    # 保存结果
    try:
        sampled_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n采样结果已保存到: {output_file}")
        print(f"最终采样记录数: {len(sampled_df)}")
    except Exception as e:
        print(f"保存文件失败: {e}")

if __name__ == "__main__":
    main()
