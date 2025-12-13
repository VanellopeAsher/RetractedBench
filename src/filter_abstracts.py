import pandas as pd
import os

def filter_short_abstracts(input_file, output_file=None, min_length=50):
    """
    删除Abstract列内容小于指定字符数或Finding为空的条目
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径，如果为None则覆盖原文件
        min_length (int): Abstract最小字符数，默认50
    """
    # 读取CSV文件
    df = pd.read_csv(input_file, encoding='utf-8')
    
    print(f"原始数据行数: {len(df)}")
    
    # 统计Abstract列长度小于50字符的条目
    short_abstracts = df[df['Abstract'].str.len() < min_length]
    print(f"Abstract长度小于{min_length}字符的条目数: {len(short_abstracts)}")
    
    # 统计Finding为空的条目
    empty_findings = df[df['Finding'].isna() | (df['Finding'].str.strip() == '')]
    print(f"Finding为空的条目数: {len(empty_findings)}")
    
    # 统计需要删除的条目（Abstract不足50字符 OR Finding为空）
    to_remove = df[(df['Abstract'].str.len() < min_length) | (df['Finding'].isna()) | (df['Finding'].str.strip() == '')]
    print(f"需要删除的条目总数: {len(to_remove)}")
    
    # 显示一些需要删除的条目示例
    if len(to_remove) > 0:
        print("\n需要删除的条目示例:")
        for i, row in to_remove.head(5).iterrows():
            abstract_len = len(str(row['Abstract'])) if pd.notna(row['Abstract']) else 0
            finding_status = "空" if pd.isna(row['Finding']) or str(row['Finding']).strip() == '' else "有内容"
            print(f"- Abstract长度{abstract_len}, Finding状态: {finding_status}")
            print(f"  标题: {str(row['PaperTitle'])[:80]}...")
    
    # 过滤掉不符合条件的条目（保留Abstract>=50字符 AND Finding不为空的条目）
    filtered_df = df[(df['Abstract'].str.len() >= min_length) & 
                     (df['Finding'].notna()) & 
                     (df['Finding'].str.strip() != '')]
    
    print(f"\n过滤后数据行数: {len(filtered_df)}")
    print(f"删除了 {len(df) - len(filtered_df)} 条记录")
    
    # 保存结果
    if output_file is None:
        output_file = input_file
    
    # 创建备份
    backup_file = input_file.replace('.csv', '_backup.csv')
    df.to_csv(backup_file, index=False, encoding='utf-8')
    print(f"原文件已备份为: {backup_file}")
    
    # 保存过滤后的数据
    filtered_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"过滤后的数据已保存到: {output_file}")
    
    return filtered_df

if __name__ == "__main__":
    input_file = "matched_papers.csv"
    
    if os.path.exists(input_file):
        filtered_df = filter_short_abstracts(input_file, min_length=50)
    else:
        print(f"错误: 找不到文件 {input_file}")
