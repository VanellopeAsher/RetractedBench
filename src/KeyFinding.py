import pandas as pd
import json
import os
import time
import re # 导入正则表达式模块，用于清理JSON字符串
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import LLM # 假设您的LLM实例来自这个utils模块

# --- 全局设置与初始化 ---
# 1. 在程序开始时只初始化一次LLM实例，避免重复创建，极大提高性能。
# 2. 将模型和平台作为可配置的变量。
LLM_PLATFORM = 'openai'
LLM_MODEL_NAME = 'gpt-4o'
# *** 注意：请确保您的 'utils.py' 文件和 LLM 类已正确配置 ***
LLM_INSTANCE = LLM(model_name=LLM_MODEL_NAME, platform=LLM_PLATFORM)


def extract_key_findings(title: str, abstract: str):
    """
    使用全局LLM实例，根据优化后的Prompt提取论文摘要的关键发现。
    返回一个包含(原始响应, highlights列表)的元组，如果因网络等原因彻底失败则返回(None, None)。
    """
    system_message = "You are an expert academic editor with a keen eye for identifying the unique, citable facts from a research paper."
    
    prompt = f"""
    As an expert academic editor, extract the paper's 1–3 core, atomic findings from the provided Title and Abstract.

    Input variables passed into the prompt:

    * Title: {title}
    * Abstract: {abstract}

    Non-negotiable rules (read carefully):

    1.  **Atomic proposition definition:** Each proposition must be a single, self-contained factual claim (subject + relation + object).  Do NOT combine multiple claims with conjunctions (no “and”, no commas linking outcomes).
    2.  **Quantity:** Return exactly one atomic proposition that captures the paper’s primary novelty (subject + relation + object).
    4.  **No vague comparative adjectives:** Do NOT use words like “essential”, “effective”, “efficient”, “superior”, “improved”, “reduced” **without numeric evidence**.  If the abstract contains a numeric metric (e.g., %, fold‑change, N, p, CI), include the exact number in the proposition.
    5.  **No invention:** Never invent or estimate numbers or statistics.
    6.  **Concision & terminology:** Prefer domain-appropriate technical terms taken directly from the abstract.
    7.  **Objective factual phrasing:** Write statements as facts (present tense), avoid attributional phrasing such as "This paper shows" or "The authors argue".
    8.  **Format-only output:** Output **only** in the exact format below — nothing else (no JSON, no comments, no metadata).

    Exact output format (must match exactly):
    Finding: <atomic proposition>.
    """
    
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        response_str = ""
        try:
            response_str = LLM_INSTANCE.generate(
                prompt=messages, 
                temperature=0.1
            )
            
            # 优雅地处理API的空响应
            if not response_str or not response_str.strip():
                # API成功返回了一个空响应。这不应被视为错误，而是“没有找到发现”。
                tqdm.write(f"处理 '{title[:30]}...' 时API返回空响应，视为无发现。")
                return response_str, [] # 返回原始响应和空列表

            # 新的解析逻辑，用于处理 "Finding: <内容>" 格式
            clean_response = response_str.strip()
            finding_prefix = "Finding: "
            highlights = []
            if clean_response.startswith(finding_prefix):
                # 提取前缀后面的文字
                finding_text = clean_response[len(finding_prefix):].strip()
                if finding_text: # 确保剥离后不是空的
                    highlights.append(finding_text)
            elif clean_response:
                # 如果没有前缀但有内容，作为备用方案，将整个响应视为finding
                # 这可以应对模型偶尔不完全遵守格式指令的情况
                tqdm.write(f"处理 '{title[:30]}...' 时响应格式不符: 未找到 'Finding: ' 前缀。将使用完整响应。")
                highlights.append(clean_response)

            return response_str, highlights # 返回原始响应和解析后的列表
        
        except Exception as e:
            tqdm.write(f"处理 '{title[:30]}...' 时发生错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    
    tqdm.write(f"处理 '{title[:30]}...' 彻底失败，将在下次运行时重试。")
    return None, None

def process_and_update_file(input_file, max_workers=4, save_interval=20):
    """
    在原始CSV文件中直接添加Finding栏，并支持断点续传。
    会定期将进度保存回文件。
    """
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return

    required_columns = ['PaperDOI', 'PaperTitle', 'Abstract']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"输入文件中缺少必要的字段。请确保文件包含: {', '.join(required_columns)}")

    # --- 断点续传机制 ---
    # 如果Finding栏不存在，创建它并填充为NA（代表未处理）。
    # 这与空字符串（""，代表已处理但无发现）区分开来。
    if 'Finding' not in df.columns:
        df['Finding'] = pd.NA
    
    # --- 筛选需要处理的任务 ---
    tasks = []
    # 只选择 Finding 栏为 NA (Not Available) 的行进行处理
    for index, row in df.iterrows():
        is_unprocessed = pd.isna(row['Finding'])
        is_valid_abstract = pd.notna(row['Abstract']) and isinstance(row['Abstract'], str) and len(row['Abstract']) > 50
        is_valid_title = pd.notna(row['PaperTitle']) and isinstance(row['PaperTitle'], str)
        
        if is_unprocessed and is_valid_abstract and is_valid_title:
            tasks.append(row)
            
    if not tasks:
        print("没有新的有效数据需要处理。")
        return
        
    print(f"总共有 {len(tasks)} 笔新数据需要通过LLM处理。")

    # --- 并行处理与定期保存 ---
    batch_results = {} # 用于暂存一个批次的结果
    total_processed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(extract_key_findings, row['PaperTitle'], row['Abstract']): row for row in tasks}
        
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="处理论文摘要"):
            original_row = future_to_task[future]
            doi = original_row['PaperDOI']
            try:
                raw_response, findings_list = future.result()
                
                if raw_response is not None:
                    tqdm.write(f"\n--- Raw Response for '{original_row.get('PaperTitle', '')[:40]}...' ---\n{raw_response}\n--------------------")

                if findings_list is not None: # API调用成功
                    finding_text = findings_list[0] if findings_list else ""
                    batch_results[doi] = finding_text
                    total_processed += 1
                
                    # 当暂存的结果达到保存间隔时，执行一次写入操作
                    if len(batch_results) >= save_interval:
                        tqdm.write(f"\n--- 达到保存点，正在更新并保存 {len(batch_results)} 条新记录... ---")
                        for temp_doi, temp_finding in batch_results.items():
                            df.loc[df['PaperDOI'] == temp_doi, 'Finding'] = temp_finding
                        
                        try:
                            df.to_csv(input_file, index=False, encoding='utf-8')
                            tqdm.write(f"--- 进度已保存。 ---\n")
                        except Exception as e:
                            tqdm.write(f"\n--- 定期保存文件时出错: {e} ---\n")
                        
                        # 清空批次，为下一批做准备
                        batch_results.clear()

            except Exception as e:
                tqdm.write(f"一个任务在处理时发生了无法恢复的错误: {e} | 处理DOI: {doi}")

    # --- 最终保存 ---
    # 处理循环结束后剩余的不足一个批次的结果
    if batch_results:
        print(f"处理完成，正在保存剩余的 {len(batch_results)} 笔记录...")
        for doi, finding in batch_results.items():
            df.loc[df['PaperDOI'] == doi, 'Finding'] = finding
        
    if total_processed > 0:
        # 在最终保存前，将所有NA值填充为空字符串，确保文件一致性
        df['Finding'] = df['Finding'].fillna('')
        
        try:
            df.to_csv(input_file, index=False, encoding='utf-8')
            print(f"文件 {input_file} 已成功更新。")
        except Exception as e:
            print(f"错误：无法写回文件 {input_file}。错误信息: {e}")
            backup_file = f"backup_{os.path.basename(input_file)}"
            df.to_csv(backup_file, index=False, encoding='utf-8')
            print(f"数据已保存至备份文件: {backup_file}")
    else:
        print("本次运行没有需要更新的数据。")


if __name__ == "__main__":
    input_file = "matched_papers.csv"
    
    # 建议的并行工作线程数，可以根据您的机器性能和API速率限制进行调整
    # 如果遇到API速率限制错误，请尝试降低此数值
    process_and_update_file(input_file, max_workers=16, save_interval=20)

