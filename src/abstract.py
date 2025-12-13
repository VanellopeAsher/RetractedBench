import pandas as pd
import requests
import time
from tqdm import tqdm
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import xml.etree.ElementTree as ET

# --- 配置 ---
# 输入和输出的CSV文件路径 (脚本会读取此文件，添加摘要后再写回)
CSV_PATH = 'matched_papers.csv'
# 您的邮箱地址，用于OpenAlex API的礼貌池
YOUR_EMAIL = "vanellopeasher1216@gmail.com"

# --- 高性能并发配置 ---
NUM_WORKERS = 16  # 并发处理的线程数量
# 为不同API设置不同的速率上限
GENERAL_API_REQUESTS_PER_SECOND = 9  # Crossref 和 OpenAlex 的速率
PUBMED_API_REQUESTS_PER_SECOND = 3   # PubMed 的安全速率 (官方建议 <= 3)
API_BATCH_SIZE = 50 # 每次API调用包含的DOI数量 (OpenAlex上限为50)
CHUNK_SIZE = 1000 # 每次向线程池提交的任务批次大小，也是保存文件的频率
VERBOSE_LOGGING = True # 新增：是否显示详细的过程日志

# --- 速率控制器 (全新、更稳健的实现) ---

class RateLimiter:
    """一个简单、线程安全的速率控制器，通过强制间隔来平滑请求突发。"""
    def __init__(self, requests_per_second):
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.last_call_time = 0

    def wait(self):
        with self.lock:
            current_time = time.monotonic()
            elapsed = current_time - self.last_call_time
            
            wait_time = self.interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            
            self.last_call_time = time.monotonic()

# --- 核心网络请求函数 ---

def make_request(url, session, rate_limiter, method='get', params=None, headers=None, timeout=60):
    """一个健壮的、通用的网络请求函数，内置完整的重试逻辑。"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [403, 429] and attempt < max_retries - 1:
                error_code = e.response.status_code
                wait_time = 30 * (attempt + 1)
                tqdm.write(f"\n  [!] 请求 {url[:60]}... 收到 {error_code}。等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
                continue
            return e.response 
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                return None

# --- 辅助函数 ---

def deconstruct_abstract(inverted_index):
    """将OpenAlex的倒排索引格式摘要转换回纯文本字符串。"""
    if not inverted_index:
        return None
    try:
        max_index = max(pos for positions in inverted_index.values() for pos in positions)
        abstract_list = [""] * (max_index + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                abstract_list[pos] = word
        return " ".join(filter(None, abstract_list))
    except Exception:
        return None

def get_abstracts_by_dois_batch(doi_batch, session, rate_limiter):
    """Source 3: 通过一批DOI从OpenAlex批量获取论文的摘要。"""
    if not doi_batch:
        return {}
    doi_filter = "|".join(filter(None, doi_batch))
    url = f"https://api.openalex.org/works?filter=doi:{doi_filter}&select=doi,abstract_inverted_index&per-page={len(doi_batch)}"
    headers = {'User-Agent': f'AbstractBatchFetcher/1.0 (mailto:{YOUR_EMAIL})'}
    
    response = make_request(url, session, rate_limiter, headers=headers)
    
    if response and response.status_code == 200:
        try:
            data = response.json()
            results = data.get('results', [])
            abstracts_map = {}
            for work in results:
                work_doi = str(work.get('doi')).replace("https://doi.org/", "")
                inverted_abstract = work.get('abstract_inverted_index')
                abstract = deconstruct_abstract(inverted_abstract)
                abstracts_map[work_doi] = abstract if abstract else "Abstract Not Available"
            for doi in doi_batch:
                if doi not in abstracts_map:
                    abstracts_map[doi] = "Abstract Not Found"
            return abstracts_map
        except Exception:
            return {doi: "Batch JSON Parse Error" for doi in doi_batch}
    
    status = response.status_code if response is not None else "Connection"
    return {doi: f"Batch HTTP Error: {status}" for doi in doi_batch}


def get_abstract_from_crossref(doi, session, rate_limiter):
    """Source 1: 从Crossref API获取摘要。"""
    url = f"https://api.crossref.org/works/{doi}"
    headers = {'User-Agent': f'AbstractFetcher/1.0 (mailto:{YOUR_EMAIL})'}
    response = make_request(url, session, rate_limiter, headers=headers)
    
    if response and response.status_code == 200:
        try:
            data = response.json()
            abstract_html = data.get("message", {}).get("abstract")
            if abstract_html:
                clean_abstract = re.sub('<[^<]+?>', '', abstract_html).strip()
                if clean_abstract:
                    return clean_abstract
            # 如果请求成功但没有摘要，可以选择性地记录
            if VERBOSE_LOGGING:
                tqdm.write(f"  [-] Crossref: DOI {doi} 查询成功，但无摘要。将尝试下一数据源。")
        except Exception:
            pass
    
    elif response is None:
        tqdm.write(f"  [-] Crossref: DOI {doi} 查询失败 (连接错误)。将尝试下一数据源。")
    elif response.status_code not in [200, 404]:
        tqdm.write(f"  [-] Crossref: DOI {doi} 查询失败，状态码: {response.status_code}。将尝试下一数据源。")
        
    return None

def get_abstract_from_pubmed(doi, session, rate_limiter):
    """Source 2: 从PubMed API获取摘要。"""
    headers = {'User-Agent': f'AbstractFetcher/1.0 (mailto:{YOUR_EMAIL})'}
    
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {'db': 'pubmed', 'term': doi, 'retmode': 'json'}
    search_response = make_request(search_url, session, rate_limiter, params=search_params, headers=headers)

    if not (search_response and search_response.status_code == 200):
        if search_response is None:
            tqdm.write(f"  [-] PubMed (Search): DOI {doi} 查询失败 (连接错误)。")
        elif search_response.status_code not in [200, 404]:
             tqdm.write(f"  [-] PubMed (Search): DOI {doi} 查询失败，状态码: {search_response.status_code}。")
        return None
    
    try:
        search_data = search_response.json()
        pmid_list = search_data.get("esearchresult", {}).get("idlist")
        if not pmid_list:
            if VERBOSE_LOGGING:
                tqdm.write(f"  [-] PubMed: DOI {doi} 查询成功，但未找到对应PMID。")
            return None
        pmid = pmid_list[0]
    except Exception:
        return None
        
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {'db': 'pubmed', 'id': pmid, 'rettype': 'abstract', 'retmode': 'xml'}
    fetch_response = make_request(fetch_url, session, rate_limiter, params=fetch_params, headers=headers)
    
    if not (fetch_response and fetch_response.status_code == 200):
        if fetch_response is None:
            tqdm.write(f"  [-] PubMed (Fetch): PMID {pmid} 查询失败 (连接错误)。")
        elif fetch_response.status_code not in [200, 404]:
            tqdm.write(f"  [-] PubMed (Fetch): PMID {pmid} 查询失败，状态码: {fetch_response.status_code}。")
        return None
        
    try:
        root = ET.fromstring(fetch_response.content)
        abstract_texts = [node.text for node in root.findall('.//AbstractText') if node.text]
        if abstract_texts:
            return " ".join(abstract_texts)
    except Exception:
        return None
    return None

# --- 工作线程执行的函数 ---
def worker(batch_data, session, general_rate_limiter, pubmed_rate_limiter):
    """
    工作线程函数：实现三级降级策略 (Crossref -> PubMed -> OpenAlex)。
    为不同API使用不同的速率控制器。
    """
    final_results = {}
    dois_for_openalex = {} # Maps index to DOI for the final batch call

    for index, doi in batch_data.items():
        if not doi or pd.isna(doi):
            final_results[index] = "DOI Invalid"
            continue

        abstract = get_abstract_from_crossref(doi, session, general_rate_limiter)
        if abstract:
            final_results[index] = abstract
            continue

        abstract = get_abstract_from_pubmed(doi, session, pubmed_rate_limiter)
        if abstract:
            final_results[index] = abstract
            continue
        
        dois_for_openalex[index] = doi

    if dois_for_openalex:
        doi_to_index_map = {doi: index for index, doi in dois_for_openalex.items()}
        openalex_results_by_doi = get_abstracts_by_dois_batch(list(doi_to_index_map.keys()), session, general_rate_limiter)
        for doi, abstract in openalex_results_by_doi.items():
            original_index = doi_to_index_map.get(doi)
            if original_index is not None:
                final_results[original_index] = abstract

    return final_results

# --- 主逻辑 ---
def main():
    """主函数，用于读取数据、爬取摘要并更新文件。"""
    print("--- 开始运行摘要爬取脚本 (高性能批处理 + 降级策略模式) ---")
    
    if YOUR_EMAIL == "your_email@example.com":
        print("\n\n致命错误: 请在脚本顶部将 'YOUR_EMAIL' 变量更改为您自己的真实邮箱地址。")
        return

    if not os.path.exists(CSV_PATH):
        print(f"错误: 输入文件 '{CSV_PATH}' 未找到。")
        return

    try:
        # 尝试多种编码格式读取CSV文件
        encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1', 'cp1252']
        df = None
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取 {len(df)} 条记录。")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("尝试了多种编码格式都无法读取文件，请检查文件编码。")
            return
            
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    if 'Abstract' not in df.columns:
        df['Abstract'] = pd.NA
    
    df['Abstract'] = df['Abstract'].replace('Abstract Not Available', pd.NA)
    rows_to_process = df[df['Abstract'].isna()]
    
    if rows_to_process.empty:
        print("所有条目均已有摘要。无需执行任何操作。")
        return
        
    print(f"共找到 {len(rows_to_process)} 个条目需要爬取摘要。")

    session = requests.Session()
    general_rate_limiter = RateLimiter(GENERAL_API_REQUESTS_PER_SECOND)
    pubmed_rate_limiter = RateLimiter(PUBMED_API_REQUESTS_PER_SECOND)

    indices_to_process = rows_to_process.index.tolist()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        with tqdm(total=len(rows_to_process), desc="爬取摘要") as pbar:
            for i in range(0, len(indices_to_process), CHUNK_SIZE):
                chunk_indices = indices_to_process[i:i + CHUNK_SIZE]
                
                api_batches = []
                for j in range(0, len(chunk_indices), API_BATCH_SIZE):
                    batch_indices = chunk_indices[j:j + API_BATCH_SIZE]
                    batch_data = {index: df.loc[index, 'PaperDOI'] for index in batch_indices}
                    api_batches.append(batch_data)

                futures = {
                    executor.submit(worker, api_batch, session, general_rate_limiter, pubmed_rate_limiter): api_batch
                    for api_batch in api_batches
                }
                
                for future in as_completed(futures):
                    try:
                        results_dict = future.result()
                        for index, abstract in results_dict.items():
                            df.loc[index, 'Abstract'] = abstract
                        
                        pbar.update(len(results_dict))
                    except Exception as exc:
                        failed_batch = futures[future]
                        pbar.write(f"\n[!] 处理一个批次时出现严重错误: {exc}")
                        for index in failed_batch.keys():
                            df.loc[index, 'Abstract'] = f"Error: {exc}"
                        pbar.update(len(failed_batch))

                try:
                    pbar.write(f"\n... 任务块 {i//CHUNK_SIZE + 1} 处理完成，正在将进度写回CSV文件 ...")
                    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
                except Exception as e:
                    pbar.write(f"\n[!] 在保存点写入CSV时出错: {e}")

    print("\n... 所有批次处理完毕 ...")
    try:
        df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"处理完成！最终结果已更新到 '{CSV_PATH}'。")
    except Exception as e:
        print(f"\n[!] 写入最终文件时出错: {e}")

if __name__ == "__main__":
    main()

