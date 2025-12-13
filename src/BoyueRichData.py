import time
import re
import json
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool, Manager
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import pickle
from datetime import datetime

# 加载.env文件
load_dotenv()

# --- 直接使用OpenAI客户端 ---
LLM_MODEL_NAME = 'gemini-2.5-pro' 

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv('CUSTOM_API_KEY'),
    base_url=os.getenv('CUSTOM_API_ENDPOINT')
)

# 模型价格配置 (每1000 tokens的价格，单位：美元)
MODEL_PRICING = {
    "deepseek-ai/DeepSeek-V3": {
        "input": 0.000282,    # SiliconFlow pricing: ¥2/M tokens ≈ $0.28/M tokens ≈ $0.00028/1K tokens
        "output": 0.001127    # SiliconFlow pricing: ¥8/M tokens ≈ $1.12/M tokens ≈ $0.00112/1K tokens
    },
    "deepseek-ai/DeepSeek-R1": {
        "input": 0.000564,    # SiliconFlow pricing: ¥4/M tokens
        "output": 0.002254    # SiliconFlow pricing: ¥16/M tokens
    },
    "Qwen/Qwen3-235B-A22B": {
        "input": 0.000352,    # SiliconFlow pricing: ¥2.5/M tokens
        "output": 0.001409    # SiliconFlow pricing: ¥10/M tokens
    },
    "gemini-2.5-pro": {
        "input": 0.00125,    # Gemini-2.5-Pro: $1.25/M tokens = $0.00125/1K tokens
        "output": 0.01000    # Gemini-2.5-Pro: $10.00/M tokens = $0.01000/1K tokens
    },
    "claude-opus-4-1-20250805": {
        "input": 0.0015, 
        "output": 0.0075
    },
    "claude-sonnet-4-20250514": {
        "input": 0.0003, 
        "output": 0.0015
    },
    "gpt-4o-2024-08-06": {
        "input": 0.00025, 
        "output": 0.0010
    },
    "o3-deep-research": {
        "input": 0.0010, 
        "output": 0.0040
    },
    "o4-mini-deep-research": {
        "input": 0.0002, 
        "output": 0.0008
    }
}

def calculate_cost(usage, model_name):
    """计算API调用成本"""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (usage.prompt_tokens / 1000) * pricing["input"]
    output_cost = (usage.completion_tokens / 1000) * pricing["output"]
    return input_cost + output_cost

# 实验统计类
class ExperimentTracker:
    def __init__(self, experiment_id: str, output_file: str = "experiment_stats.json"):
        self.experiment_id = experiment_id
        self.output_file = output_file
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = datetime.now().isoformat()
        
    def add_call(self, usage, model_name):
        """添加一次API调用记录"""
        if usage:
            self.total_calls += 1
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            
            # 计算成本
            cost = calculate_cost(usage, model_name)
            self.total_cost += cost
            
            # 实时保存到JSON文件
            self.save_stats()
    
    def save_stats(self):
        """保存统计数据到JSON文件"""
        stats = {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "last_updated": datetime.now().isoformat(),
            "total_stats": {
                "total_calls": self.total_calls,
                "total_input_token": self.total_input_tokens,
                "total_output_token": self.total_output_tokens,
                "total_token": self.total_tokens,
                "total_cost": round(self.total_cost, 6)
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def get_stats(self):
        """获取当前统计数据"""
        return {
            "experiment_id": self.experiment_id,
            "total_stats": {
                "total_calls": self.total_calls,
                "total_input_token": self.total_input_tokens,
                "total_output_token": self.total_output_tokens,
                "total_token": self.total_tokens,
                "total_cost": round(self.total_cost, 6)
            }
        }
def process_finding_worker(args):
    """多进程工作函数"""
    finding, row_data, tracker = args
    
    # 重定向输出到空
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        try:
            # 初始化OpenAI客户端（每个进程需要独立的客户端）
            worker_client = OpenAI(
                api_key=os.getenv('CUSTOM_API_KEY'),
                base_url=os.getenv('CUSTOM_API_ENDPOINT')
            )
            
            system_message = """You are a highly analytical AI research assistant. Your task is to find reliable academic sources that directly support or opposed to a given claim. You must base your decision on verifiable sources such as peer-reviewed journal articles, conference papers, or official publications."""
            
            json_structure_example = """
            {
              "analysis_summary": "A brief, one-sentence summary of your findings based on the sources you found.",
              "selected_sources": [
                {
                  "title": "The exact title of the selected paper.",
                  "doi": "The paper's DOI. Use null if not available.",
                  "source_url": "The URL where the paper can be accessed."
                }
              ]
            }
            """

            prompt = f"""
            **Claim to Verify:** "{finding}"

            **Your Task:**
            1. Search for 5 reliable, peer-reviewed source that directly supports or opposed to the claim.
            2. Evaluate their abstracts or content to ensure relevance.
            3. Only include sources that clearly support or opposed to the claim. You must find 5 reliable, real academic sources.
            4. Format your answer strictly as a single JSON object following the required structure.

            **Required JSON Structure:**
            {json_structure_example}
            """
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            completion = worker_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                temperature=0.1
            )
            
            # 跟踪API使用量
            if hasattr(completion, 'usage') and completion.usage and tracker:
                tracker.add_call(completion.usage, LLM_MODEL_NAME)
            
            response_content = completion.choices[0].message.content
            
            # 解析JSON响应
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group(0))
                
                # 创建输出行
                output_row = {
                    'RetractedRecordID': row_data['RetractedRecordID'],
                    'RetractedDOI': row_data['RetractedDOI'],
                    'RetractedPaperTitle': row_data['RetractedPaperTitle'],
                    'PaperTitle': row_data['PaperTitle'],
                    'CitationCount': row_data['CitationCount'],
                    'Finding': finding,
                    'AnalysisSummary': parsed_json.get('analysis_summary', ''),
                    'CitationSource1': '',
                    'CitationSource2': '',
                    'CitationSource3': '',
                    'CitationSource4': '',
                    'CitationSource5': ''
                }
                
                # 填充CitationSource1-5
                sources = parsed_json.get('selected_sources', [])
                for i, source in enumerate(sources[:5]):
                    citation_key = f'CitationSource{i+1}'
                    citation_text = f"Title: {source.get('title', 'N/A')}; DOI: {source.get('doi', 'N/A')}; URL: {source.get('source_url', 'N/A')}"
                    output_row[citation_key] = citation_text
                
                return output_row
            else:
                return create_failed_row(row_data, finding, "JSON parsing failed")
                
        except Exception as e:
            return create_failed_row(row_data, finding, f"Processing failed: {str(e)}")

def create_failed_row(row_data, finding, error_msg):
    """创建失败的输出行"""
    return {
        'RetractedRecordID': row_data['RetractedRecordID'],
        'RetractedDOI': row_data['RetractedDOI'],
        'RetractedPaperTitle': row_data['RetractedPaperTitle'],
        'PaperTitle': row_data['PaperTitle'],
        'CitationCount': row_data['CitationCount'],
        'Finding': finding,
        'AnalysisSummary': error_msg,
        'CitationSource1': '',
        'CitationSource2': '',
        'CitationSource3': '',
        'CitationSource4': '',
        'CitationSource5': ''
    }


def process_csv_file(csv_file_path, output_file_path, experiment_id="LLM-test-001"):
    """处理CSV文件中的所有条目（只处理AnalysisSummary为空的条目）"""
    
    # 初始化实验跟踪器
    tracker = ExperimentTracker(experiment_id, f"experiment_{experiment_id}.json")
    print(f"开始实验: {experiment_id}")
    print(f"统计文件: {tracker.output_file}")
    
    # 读取原始CSV文件
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file_path, encoding='latin-1')
        except:
            df = pd.read_csv(csv_file_path, encoding='cp1252')
    
    # 检查是否已经存在输出文件
    existing_results = []
    if os.path.exists(output_file_path):
        try:
            existing_df = pd.read_csv(output_file_path, encoding='utf-8')
            existing_results = existing_df.to_dict('records')
            print(f"\n发现已存在的输出文件，已有 {len(existing_results)} 条记录")
        except:
            existing_results = []
    
    # 创建结果字典，以RetractedRecordID为键
    results_dict = {}
    processed_count = 0
    
    # 加载已存在的结果
    for result in existing_results:
        record_id = result.get('RetractedRecordID')
        if record_id:
            results_dict[record_id] = result
            # 检查是否已处理过
            analysis_summary = result.get('AnalysisSummary')
            if analysis_summary and isinstance(analysis_summary, str) and analysis_summary.strip():
                processed_count += 1
    
    # 准备需要处理的工作项目（只处理AnalysisSummary为空的）
    work_items = []
    for index, row in df.iterrows():
        record_id = row['RetractedRecordID']
        
        # 检查是否已经处理过
        existing_result = results_dict.get(record_id)
        if existing_result:
            analysis_summary = existing_result.get('AnalysisSummary')
            if analysis_summary and isinstance(analysis_summary, str) and analysis_summary.strip():
                continue  # 跳过已处理的记录
        
        finding = row['Finding']
        row_data = {
            'RetractedRecordID': record_id,
            'RetractedDOI': row['RetractedDOI'],
            'RetractedPaperTitle': row['RetractedPaperTitle'],
            'PaperTitle': row['PaperTitle'],
            'CitationCount': row['CitationCount']
        }
        work_items.append((finding, row_data, tracker))
    
    if not work_items:
        print("所有记录已处理完成！")
        return
    
    print(f"\n开始处理 {len(work_items)} 条未处理记录（已处理: {processed_count}）")
    
    batch_size = 20
    
    with Pool(processes=4) as pool:
        # 使用tqdm显示进度
        with tqdm(total=len(work_items), desc="Processing", unit="records") as pbar:
            
            # 分批处理
            for i in range(0, len(work_items), batch_size):
                batch = work_items[i:i + batch_size]
                
                try:
                    # 处理当前批次
                    batch_results = pool.map(process_finding_worker, batch)
                    
                    # 更新结果字典
                    for result in batch_results:
                        record_id = result.get('RetractedRecordID')
                        if record_id:
                            results_dict[record_id] = result
                    
                    # 更新进度条
                    pbar.update(len(batch))
                    
                    # 每20条记录保存一次（追加新结果）
                    new_results = batch_results
                    if new_results:
                        new_df = pd.DataFrame(new_results)
                        # 检查文件是否存在，决定是否写入表头
                        write_header = not os.path.exists(output_file_path)
                        new_df.to_csv(output_file_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
                    
                except KeyboardInterrupt:
                    print("\n检测到中断，保存当前结果...")
                    if batch_results:
                        new_df = pd.DataFrame(batch_results)
                        write_header = not os.path.exists(output_file_path)
                        new_df.to_csv(output_file_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
                    raise
                except Exception as e:
                    print(f"\n处理批次时发生错误: {e}")
                    continue
    
    # 处理完成，不需要最终保存（已经通过追加方式保存）
    print(f"\n处理完成！所有新结果已追加保存到: {output_file_path}")
    
    # 显示最终统计
    print("\n" + "="*60)
    print("实验完成 - 最终统计")
    print("="*60)
    final_stats = tracker.get_stats()
    print(json.dumps(final_stats, indent=2, ensure_ascii=False))
    print(f"统计数据已保存到: {tracker.output_file}")

def test_single_finding():
    """测试单个研究发现的验证"""
    # 初始化实验跟踪器
    tracker = ExperimentTracker("LLM-test-001", "test_experiment_stats.json")
    
    finding = "Substitution of Cd2+ in Sr1-xCdxZn2Fe4O11 R-type hexaferrites increases saturation magnetization (Ms) from 49.76 to 56.38 emu/g, retentivity (Mr) from 15.82 to 18.30 emu/g, and coercivity (Hc) from 203.20 Oe to 215.80 Oe."
    row_data = {
        'RetractedRecordID': 'TEST001',
        'RetractedDOI': 'test.doi',
        'RetractedPaperTitle': 'Test Paper',
        'PaperTitle': 'Test Matched Paper',
        'CitationCount': 10
    }
    result = process_finding_worker((finding, row_data, tracker))
    
    # 显示统计信息
    print("\n" + "="*50)
    print("测试完成 - 统计信息")
    print("="*50)
    stats = tracker.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    return result

if __name__ == "__main__":
    
    csv_file = r'c:\Users\MateBookGT14\OneDrive\桌面\retracted\optimization_25.8.18\sampled_matched_papers_1k.csv'
    output_file = rf'c:\Users\MateBookGT14\OneDrive\桌面\retracted\optimization_25.8.18\enriched_papers_with_gemini-2.5-pro.csv'
    
    if os.path.exists(csv_file):
        process_csv_file(csv_file, output_file, "LLM-test-001")
    else:
        print(f"文件不存在: {csv_file}")
        print("运行单个测试用例...")
        test_single_finding()

