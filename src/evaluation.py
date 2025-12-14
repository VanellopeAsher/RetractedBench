#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py - 直接读取输入文件的HIT_MISS字段，输出hit rate统计结果
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path

class HitRateEvaluator:
    def __init__(self):
        self.results = {}
    
    def load_data(self, file_path):
        """加载数据文件"""
        print(f"正在加载文件: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 尝试多种编码方式读取CSV文件
        encodings = ['utf-8', 'latin-1', 'cp1252', 'gbk']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                print(f"使用 {encoding} 编码失败，尝试下一个...")
                continue
        
        if df is None:
            raise ValueError("无法使用任何编码读取CSV文件")
        
        print(f"数据加载完成，总记录数: {len(df)}")
        return df
    
    def preprocess_data(self, df):
        """预处理数据"""
        print("正在预处理数据...")
        
        # 检查必要的字段
        required_fields = ['HIT_MISS', 'Retracted_Status']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            raise ValueError(f"缺少必要字段: {missing_fields}")
        
        # 检查是否有CitationSource验证字段
        citation_valid_columns = ['CitationSource1_valid', 'CitationSource2_valid', 'CitationSource3_valid', 'CitationSource4_valid', 'CitationSource5_valid']
        has_citation_valid_fields = any(col in df.columns for col in citation_valid_columns)
        if has_citation_valid_fields:
            print("检测到CitationSource验证字段，将计算相关指标")
        else:
            print("未检测到CitationSource验证字段，跳过相关指标计算")
        
        # 删除HIT_MISS为空或无效的记录
        initial_count = len(df)
        df = df.dropna(subset=['HIT_MISS'])
        df = df[df['HIT_MISS'].isin(['HIT', 'MISS', 'Not Verified'])]
        filtered_count = len(df)
        
        if initial_count != filtered_count:
            print(f"过滤掉 {initial_count - filtered_count} 条无效记录")
        
        # 创建IS_RETRACTED字段（基于Retracted_Status: yes -> 撤稿, no -> 非撤稿）
        status_series = df['Retracted_Status'].astype(str).str.strip().str.lower()
        df['IS_RETRACTED'] = status_series.eq('yes')
        
        # 不创建新字段，直接计算
        
        print(f"预处理完成，有效记录数: {len(df)}")
        print(f"撤稿论文: {len(df[df['IS_RETRACTED']==True])}")
        print(f"非撤稿论文: {len(df[df['IS_RETRACTED']==False])}")
        
        return df, has_citation_valid_fields
    
    def calculate_hit_rates(self, df, has_citation_valid_fields=False):
        """计算各种HIT率和CitationSource验证比例"""
        print("正在计算HIT率和CitationSource验证比例...")
        
        # 整体HIT率
        overall_hit_rate = (df['HIT_MISS'] == 'HIT').mean()
        overall_count = len(df)
        overall_hits = (df['HIT_MISS'] == 'HIT').sum()
        
        # 撤稿论文HIT率
        retracted_df = df[df['IS_RETRACTED'] == True]
        retracted_hit_rate = (retracted_df['HIT_MISS'] == 'HIT').mean() if len(retracted_df) > 0 else 0
        retracted_count = len(retracted_df)
        retracted_hits = (retracted_df['HIT_MISS'] == 'HIT').sum() if len(retracted_df) > 0 else 0
        
        # 非撤稿论文HIT率
        non_retracted_df = df[df['IS_RETRACTED'] == False]
        non_retracted_hit_rate = (non_retracted_df['HIT_MISS'] == 'HIT').mean() if len(non_retracted_df) > 0 else 0
        non_retracted_count = len(non_retracted_df)
        non_retracted_hits = (non_retracted_df['HIT_MISS'] == 'HIT').sum() if len(non_retracted_df) > 0 else 0
        
        # CitationSource验证比例（如果字段存在）
        if has_citation_valid_fields:
            # 计算所有CitationSource验证字段的yes比例
            citation_valid_columns = ['CitationSource1_valid', 'CitationSource2_valid', 'CitationSource3_valid', 'CitationSource4_valid', 'CitationSource5_valid']
            existing_columns = [col for col in citation_valid_columns if col in df.columns]
            
            if existing_columns:
                # 整体CitationSource验证比例
                overall_valid_count = 0
                overall_total_count = 0
                for col in existing_columns:
                    valid_mask = df[col] == 'yes'
                    overall_valid_count += valid_mask.sum()
                    overall_total_count += df[col].notna().sum()
                overall_non_hallucination_rate = overall_valid_count / overall_total_count if overall_total_count > 0 else 0
                overall_non_hallucination_count = overall_valid_count
                
                # 撤稿论文CitationSource验证比例
                retracted_valid_count = 0
                retracted_total_count = 0
                for col in existing_columns:
                    valid_mask = (retracted_df[col] == 'yes') if len(retracted_df) > 0 else pd.Series([False] * len(retracted_df))
                    retracted_valid_count += valid_mask.sum()
                    retracted_total_count += retracted_df[col].notna().sum() if len(retracted_df) > 0 else 0
                retracted_non_hallucination_rate = retracted_valid_count / retracted_total_count if retracted_total_count > 0 else 0
                retracted_non_hallucination_count = retracted_valid_count
                
                # 非撤稿论文CitationSource验证比例
                non_retracted_valid_count = 0
                non_retracted_total_count = 0
                for col in existing_columns:
                    valid_mask = (non_retracted_df[col] == 'yes') if len(non_retracted_df) > 0 else pd.Series([False] * len(non_retracted_df))
                    non_retracted_valid_count += valid_mask.sum()
                    non_retracted_total_count += non_retracted_df[col].notna().sum() if len(non_retracted_df) > 0 else 0
                non_retracted_non_hallucination_rate = non_retracted_valid_count / non_retracted_total_count if non_retracted_total_count > 0 else 0
                non_retracted_non_hallucination_count = non_retracted_valid_count
            else:
                overall_non_hallucination_rate = None
                overall_non_hallucination_count = None
                retracted_non_hallucination_rate = None
                retracted_non_hallucination_count = None
                non_retracted_non_hallucination_rate = None
                non_retracted_non_hallucination_count = None
        else:
            # 如果没有CitationSource验证字段，设置为None
            overall_non_hallucination_rate = None
            overall_non_hallucination_count = None
            retracted_non_hallucination_rate = None
            retracted_non_hallucination_count = None
            non_retracted_non_hallucination_rate = None
            non_retracted_non_hallucination_count = None
        
        # 计算置信区间 (95%)
        def calculate_ci(n, p):
            if n == 0:
                return 0, 0
            se = np.sqrt(p * (1 - p) / n)
            ci_lower = max(0, p - 1.96 * se)
            ci_upper = min(1, p + 1.96 * se)
            return ci_lower, ci_upper
        
        overall_ci_lower, overall_ci_upper = calculate_ci(overall_count, overall_hit_rate)
        retracted_ci_lower, retracted_ci_upper = calculate_ci(retracted_count, retracted_hit_rate)
        non_retracted_ci_lower, non_retracted_ci_upper = calculate_ci(non_retracted_count, non_retracted_hit_rate)
        
        # 计算CitationSource验证置信区间（如果字段存在）
        if has_citation_valid_fields and overall_non_hallucination_rate is not None:
            overall_nh_ci_lower, overall_nh_ci_upper = calculate_ci(overall_total_count, overall_non_hallucination_rate)
            retracted_nh_ci_lower, retracted_nh_ci_upper = calculate_ci(retracted_total_count, retracted_non_hallucination_rate)
            non_retracted_nh_ci_lower, non_retracted_nh_ci_upper = calculate_ci(non_retracted_total_count, non_retracted_non_hallucination_rate)
        else:
            overall_nh_ci_lower = overall_nh_ci_upper = None
            retracted_nh_ci_lower = retracted_nh_ci_upper = None
            non_retracted_nh_ci_lower = non_retracted_nh_ci_upper = None
        
        results = {
            'overall': {
                'hit_rate': overall_hit_rate,
                'count': overall_count,
                'hits': overall_hits,
                'ci_lower': overall_ci_lower,
                'ci_upper': overall_ci_upper,
                'non_hallucination_rate': overall_non_hallucination_rate,
                'non_hallucination_count': overall_non_hallucination_count,
                'non_hallucination_ci_lower': overall_nh_ci_lower,
                'non_hallucination_ci_upper': overall_nh_ci_upper
            },
            'retracted': {
                'hit_rate': retracted_hit_rate,
                'count': retracted_count,
                'hits': retracted_hits,
                'ci_lower': retracted_ci_lower,
                'ci_upper': retracted_ci_upper,
                'non_hallucination_rate': retracted_non_hallucination_rate,
                'non_hallucination_count': retracted_non_hallucination_count,
                'non_hallucination_ci_lower': retracted_nh_ci_lower,
                'non_hallucination_ci_upper': retracted_nh_ci_upper
            },
            'non_retracted': {
                'hit_rate': non_retracted_hit_rate,
                'count': non_retracted_count,
                'hits': non_retracted_hits,
                'ci_lower': non_retracted_ci_lower,
                'ci_upper': non_retracted_ci_upper,
                'non_hallucination_rate': non_retracted_non_hallucination_rate,
                'non_hallucination_count': non_retracted_non_hallucination_count,
                'non_hallucination_ci_lower': non_retracted_nh_ci_lower,
                'non_hallucination_ci_upper': non_retracted_nh_ci_upper
            }
        }
        
        return results
    
    def print_results(self, results, file_name, has_citation_valid_fields=False):
        """打印结果"""
        print(f"\n{file_name} 评估结果:")
        print("-" * 50)
        
        # 获取结果数据
        overall = results['overall']
        retracted = results['retracted']
        non_retracted = results['non_retracted']
        
        # CitationSource验证比例（第一个指标）
        if has_citation_valid_fields and overall['non_hallucination_rate'] is not None:
            print(f"Non-Hallucination Rate: {overall['non_hallucination_rate']:.4f} ({overall['non_hallucination_rate']*100:.2f}%)")
            print(f"95% CI: [{overall['non_hallucination_ci_lower']:.4f}, {overall['non_hallucination_ci_upper']:.4f}]")
        else:
            print("Non-Hallucination Rate: N/A (字段不存在)")
        
        # 整体结果
        print(f"Overall Hit Rate: {overall['hit_rate']:.4f} ({overall['hit_rate']*100:.2f}%)")
        print(f"95% CI: [{overall['ci_lower']:.4f}, {overall['ci_upper']:.4f}]")
        
        # 撤稿论文结果
        print(f"Retracted Hit Rate: {retracted['hit_rate']:.4f} ({retracted['hit_rate']*100:.2f}%)")
        print(f"95% CI: [{retracted['ci_lower']:.4f}, {retracted['ci_upper']:.4f}]")
        
        # 非撤稿论文结果
        print(f"Non-Retracted Hit Rate: {non_retracted['hit_rate']:.4f} ({non_retracted['hit_rate']*100:.2f}%)")
        print(f"95% CI: [{non_retracted['ci_lower']:.4f}, {non_retracted['ci_upper']:.4f}]")
        
        # Hit Rate Difference
        if retracted['count'] > 0 and non_retracted['count'] > 0:
            hit_rate_diff = non_retracted['hit_rate'] - retracted['hit_rate']
            print(f"Hit Rate Difference: {hit_rate_diff:.4f} ({hit_rate_diff*100:.2f}%)")
        else:
            print("Hit Rate Difference: N/A (数据不足)")
        
        print("-" * 50)
    
    def save_results(self, results, file_name, output_file=None, has_citation_valid_fields=False):
        """保存结果到文件"""
        if output_file is None:
            output_file = f"{Path(file_name).stem}_hit_rate_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{file_name} 评估结果:\n")
            f.write("-" * 50 + "\n")
            
            # 获取结果数据
            overall = results['overall']
            retracted = results['retracted']
            non_retracted = results['non_retracted']
            
            # CitationSource验证比例（第一个指标）
            if has_citation_valid_fields and overall['non_hallucination_rate'] is not None:
                f.write(f"Non-Hallucination Rate: {overall['non_hallucination_rate']:.4f} ({overall['non_hallucination_rate']*100:.2f}%)\n")
                f.write(f"95% CI: [{overall['non_hallucination_ci_lower']:.4f}, {overall['non_hallucination_ci_upper']:.4f}]\n")
            else:
                f.write("Non-Hallucination Rate: N/A (字段不存在)\n")
            
            # 整体结果
            f.write(f"Overall Hit Rate: {overall['hit_rate']:.4f} ({overall['hit_rate']*100:.2f}%)\n")
            f.write(f"95% CI: [{overall['ci_lower']:.4f}, {overall['ci_upper']:.4f}]\n")
            
            # 撤稿论文结果
            f.write(f"Retracted Hit Rate: {retracted['hit_rate']:.4f} ({retracted['hit_rate']*100:.2f}%)\n")
            f.write(f"95% CI: [{retracted['ci_lower']:.4f}, {retracted['ci_upper']:.4f}]\n")
            
            # 非撤稿论文结果
            f.write(f"Non-Retracted Hit Rate: {non_retracted['hit_rate']:.4f} ({non_retracted['hit_rate']*100:.2f}%)\n")
            f.write(f"95% CI: [{non_retracted['ci_lower']:.4f}, {non_retracted['ci_upper']:.4f}]\n")
            
            # Hit Rate Difference
            if retracted['count'] > 0 and non_retracted['count'] > 0:
                hit_rate_diff = non_retracted['hit_rate'] - retracted['hit_rate']
                f.write(f"Hit Rate Difference: {hit_rate_diff:.4f} ({hit_rate_diff*100:.2f}%)\n")
            else:
                f.write("Hit Rate Difference: N/A (数据不足)\n")
            
            f.write("-" * 50 + "\n")
        
        print(f"结果已保存到: {output_file}")
    
    def evaluate_file(self, file_path, output_file=None):
        """评估单个文件"""
        try:
            # 加载数据
            df = self.load_data(file_path)
            
            # 预处理数据
            df, has_citation_valid_fields = self.preprocess_data(df)
            
            # 计算HIT率
            results = self.calculate_hit_rates(df, has_citation_valid_fields)
            
            # 打印结果
            self.print_results(results, os.path.basename(file_path), has_citation_valid_fields)
            
            # 保存结果
            if output_file:
                self.save_results(results, os.path.basename(file_path), output_file, has_citation_valid_fields)
            
            return results
            
        except Exception as e:
            print(f"评估文件时出错: {e}")
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估CSV文件中的HIT_MISS字段，计算各种HIT率')
    parser.add_argument('--input_file', help='输入CSV文件路径')
    parser.add_argument('-o', '--output', help='输出结果文件路径（可选）')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细输出')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = HitRateEvaluator()
    
    # 评估文件
    results = evaluator.evaluate_file(args.input_file, args.output)
    
    if results is None:
        sys.exit(1)
    else:
        print("\n评估完成！")
        sys.exit(0)

if __name__ == "__main__":
    main()
