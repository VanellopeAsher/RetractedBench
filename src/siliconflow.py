import time
import re
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 添加父目录到路径以导入utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import LLM 
except ImportError:
    print("错误：无法导入 'utils.py'。请确保该文件存在于脚本的同一目录或Python路径中。")
    exit()

# --- LLM实例初始化 ---
LLM_PLATFORM = 'siliconflow'  # 或者根据您的utils.py配置
LLM_MODEL_NAME = 'deepseek-ai/DeepSeek-R1'

try:
    LLM_INSTANCE = LLM(model_name=LLM_MODEL_NAME, platform=LLM_PLATFORM)
except Exception as e:
    print(f"错误：无法初始化LLM实例。请检查您的 'utils.py' 文件和API密钥配置。错误信息: {e}")
    exit()

# 模型价格配置 (每1000 tokens的价格，单位：美元)
MODEL_PRICING = {
    "deepseek-ai/DeepSeek-V3": {
        "input": 0.00028,    # SiliconFlow pricing: ¥2/M tokens ≈ $0.28/M tokens ≈ $0.00028/1K tokens
        "output": 0.00112    # SiliconFlow pricing: ¥8/M tokens ≈ $1.12/M tokens ≈ $0.00112/1K tokens
    },
    "deepseek-ai/DeepSeek-R1": {
        "input": 0.00056,    # SiliconFlow pricing: ¥4/M tokens
        "output": 0.00224    # SiliconFlow pricing: ¥16/M tokens
    },
    "Qwen/Qwen3-235B-A22B": {
        "input": 0.00035,    # SiliconFlow pricing: ¥2.5/M tokens
        "output": 0.0014    # SiliconFlow pricing: ¥10/M tokens
    }
}

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

def calculate_cost(usage, model_name):
    """计算API调用成本"""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (usage.prompt_tokens / 1000) * pricing["input"]
    output_cost = (usage.completion_tokens / 1000) * pricing["output"]
    return input_cost + output_cost
# --- 单个测试用例 ---

def test_single_finding():
    """测试单个研究发现的验证"""
    
    # 初始化实验跟踪器
    tracker = ExperimentTracker("LLM-test-001", "experiment_stats.json")
    
    # 测试用例
    finding = "Metal nanoparticles are synthesized using a green synthesis method from the fresh flowers of Clitoria ternatea."
    
    system_message = """You are a highly analytical AI research assistant. Your task is to find reliable academic sources that directly support a given claim. You must base your decision on verifiable sources such as peer-reviewed journal articles, conference papers, or official publications."""
    
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
    1. Search for 5 reliable, peer-reviewed source that directly supports the claim.
    2. Evaluate their abstracts or content to ensure relevance.
    3. Only include sources that clearly support the claim.
    4. If no reliable sources are found, return an empty `selected_sources` array.
    5. Format your answer strictly as a single JSON object following the required structure.

    **Required JSON Structure:**
    {json_structure_example}
    """
    
    print("="*60)
    print(f"模型: {LLM_MODEL_NAME}")
    print(f"测试Finding: {finding}")
    print("="*60)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 使用LLM_INSTANCE.generate()方法调用
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        raw_response_str = LLM_INSTANCE.generate(
            prompt=messages,
            temperature=0.0,
            web_search=False
        )
        
        # 计算运行时间
        end_time = time.time()
        runtime = end_time - start_time
        
        # 获取响应内容
        response_content = raw_response_str
        
        # 调试信息：检查last_usage的状态
        print(f"\n调试信息:")
        print(f"LLM_INSTANCE.last_usage 是否存在: {hasattr(LLM_INSTANCE, 'last_usage')}")
        print(f"LLM_INSTANCE.last_usage 值: {LLM_INSTANCE.last_usage}")
        if hasattr(LLM_INSTANCE, 'last_usage') and LLM_INSTANCE.last_usage:
            print(f"Usage 对象类型: {type(LLM_INSTANCE.last_usage)}")
            print(f"Usage 对象属性: {dir(LLM_INSTANCE.last_usage)}")
        
        # 尝试获取token使用量（如果LLM_INSTANCE支持）
        if hasattr(LLM_INSTANCE, 'last_usage') and LLM_INSTANCE.last_usage:
            usage = LLM_INSTANCE.last_usage
            cost = calculate_cost(usage, LLM_MODEL_NAME)
            
            # 添加到实验跟踪器
            tracker.add_call(usage, LLM_MODEL_NAME)
            
            print("\n" + "="*50)
            print("API调用统计")
            print("="*50)
            print(f"输入Token: {usage.prompt_tokens:,}")
            print(f"输出Token: {usage.completion_tokens:,}")
            print(f"总Token: {usage.total_tokens:,}")
            print(f"成本: ${cost:.6f} (SiliconFlow平台)")
            print(f"成本: ¥{cost * 7.2:.6f} (人民币)")
            print(f"运行时间: {runtime:.2f}秒")
            print("="*50)
            
            # 显示累计统计
            print("\n" + "="*50)
            print("累计实验统计")
            print("="*50)
            stats = tracker.get_stats()
            print(f"实验ID: {stats['experiment_id']}")
            print(f"总调用次数: {stats['total_stats']['total_calls']}")
            print(f"总输入Token: {stats['total_stats']['total_input_token']:,}")
            print(f"总输出Token: {stats['total_stats']['total_output_token']:,}")
            print(f"总Token: {stats['total_stats']['total_token']:,}")
            print(f"总成本: ${stats['total_stats']['total_cost']:.6f}")
            print(f"总成本: ¥{stats['total_stats']['total_cost'] * 7.2:.6f}")
            print("="*50)
        else:
            print(f"\n运行时间: {runtime:.2f}秒")
            print("警告: 无法获取token使用量信息")
        
        print("\n" + "="*50)
        print("模型响应")
        print("="*50)
        print(response_content)
        
        # 尝试解析JSON响应
        try:
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group(0))
                print("\n" + "="*50)
                print("解析后的JSON结果")
                print("="*50)
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
            else:
                print("\n警告: 响应中未找到有效的JSON格式")
        except json.JSONDecodeError as e:
            print(f"\n警告: JSON解析失败: {e}")
            
    except Exception as e:
        print(f"\n错误: API调用失败: {e}")

def create_new_experiment(experiment_id: str, output_file: str = None):
    """创建新的实验跟踪器"""
    if output_file is None:
        output_file = f"experiment_{experiment_id}.json"
    return ExperimentTracker(experiment_id, output_file)

def run_experiment_with_tracking(experiment_id: str, findings: list, output_file: str = None):
    """运行带跟踪的实验"""
    tracker = create_new_experiment(experiment_id, output_file)
    
    print(f"开始实验: {experiment_id}")
    print(f"输出文件: {tracker.output_file}")
    print("="*60)
    
    for i, finding in enumerate(findings, 1):
        print(f"\n处理第 {i}/{len(findings)} 个发现...")
        test_single_finding_with_tracker(finding, tracker)
    
    # 最终统计
    print("\n" + "="*60)
    print("实验完成 - 最终统计")
    print("="*60)
    final_stats = tracker.get_stats()
    print(json.dumps(final_stats, indent=2, ensure_ascii=False))
    print(f"统计数据已保存到: {tracker.output_file}")

def test_single_finding_with_tracker(finding: str, tracker: ExperimentTracker):
    """使用跟踪器测试单个发现"""
    system_message = """You are a highly analytical AI research assistant. Your task is to find reliable academic sources that directly support a given claim. You must base your decision on verifiable sources such as peer-reviewed journal articles, conference papers, or official publications."""
    
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
    1. Search for 5 reliable, peer-reviewed source that directly supports the claim.
    2. Evaluate their abstracts or content to ensure relevance.
    3. Only include sources that clearly support the claim.
    4. If no reliable sources are found, return an empty `selected_sources` array.
    5. Format your answer strictly as a single JSON object following the required structure.

    **Required JSON Structure:**
    {json_structure_example}
    """
    
    print(f"验证发现: {finding}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 使用LLM_INSTANCE.generate()方法调用
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        raw_response_str = LLM_INSTANCE.generate(
            prompt=messages,
            temperature=0.0,
            web_search=False
        )
        
        # 计算运行时间
        end_time = time.time()
        runtime = end_time - start_time
        
        # 尝试获取token使用量并添加到跟踪器
        if hasattr(LLM_INSTANCE, 'last_usage') and LLM_INSTANCE.last_usage:
            usage = LLM_INSTANCE.last_usage
            tracker.add_call(usage, LLM_MODEL_NAME)
            
            print(f"  - 输入Token: {usage.prompt_tokens:,}")
            print(f"  - 输出Token: {usage.completion_tokens:,}")
            print(f"  - 成本: ${calculate_cost(usage, LLM_MODEL_NAME):.6f}")
            print(f"  - 运行时间: {runtime:.2f}秒")
        else:
            print(f"  - 运行时间: {runtime:.2f}秒")
            print("  - 警告: 无法获取token使用量信息")
            
    except Exception as e:
        print(f"  - 错误: API调用失败: {e}")

if __name__ == "__main__":
    # 运行单个测试
    test_single_finding()
    
    # 示例：运行多个发现的实验
    # findings = [
    #     "Metal nanoparticles are synthesized using a green synthesis method from the fresh flowers of Clitoria ternatea.",
    #     "Graphene oxide shows excellent antibacterial properties against E. coli bacteria.",
    #     "Quantum dots can be used for targeted drug delivery in cancer treatment."
    # ]
    # run_experiment_with_tracking("LLM-test-002", findings)

