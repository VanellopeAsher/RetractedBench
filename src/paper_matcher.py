import pandas as pd
import requests
import time
from tqdm import tqdm
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import xml.etree.ElementTree as ET

# --- 配置 ---
# 输入的撤稿论文CSV文件路径
INPUT_CSV_PATH = 'retraction_watch_with_abstracts.csv'
# 输出匹配结果的CSV文件路径
OUTPUT_CSV_PATH = 'matched_papers.csv'
# 新增：输出匹配队列长度记录的文件路径
QUEUE_LENGTH_CSV_PATH = 'match_queue_lengths.csv'
# 您的邮箱地址，用于OpenAlex API的礼貌池 (https://docs.openalex.org/how-to/api/rate-limits-and-authentication)
# 这有助于OpenAlex在有问题时联系您，也是一种好的API使用实践
YOUR_EMAIL = "vanellopeasher1216@gmail.com"

# --- 高性能并发配置 ---
NUM_WORKERS = 6  # 并发处理的线程数量
API_REQUESTS_PER_SECOND = 9  # API请求速率上限 (为安全起见，略低于OpenAlex的10次/秒限制)
PUBMED_API_REQUESTS_PER_SECOND = 3   # PubMed 的安全速率 (官方建议 <= 3)

# --- 代理服务器配置 (可选) ---
PROXIES = None

# 每处理多少个条目就保存一次文件
SAVE_INTERVAL = 20 # 在高并发下，可以适当增加保存间隔

# --- 速率控制器 ---

class RateLimiter:
    """一个线程安全的速率控制器，用于确保API请求总数不超过限制。"""
    def __init__(self, rate_limit, per_seconds=1):
        self.rate_limit = rate_limit
        self.per_seconds = per_seconds
        self.allowance = float(rate_limit)
        self.last_check = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            current_time = time.monotonic()
            time_passed = current_time - self.last_check
            self.last_check = current_time
            self.allowance += time_passed * (self.rate_limit / self.per_seconds)
            if self.allowance > self.rate_limit:
                self.allowance = float(self.rate_limit)
            
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per_seconds / self.rate_limit)
                self.allowance = 0.0
            else:
                sleep_time = 0
                self.allowance -= 1.0
        
        if sleep_time > 0:
            time.sleep(sleep_time)

# --- 摘要获取相关函数 ---

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

def get_abstract_from_crossref(doi, session, rate_limiter):
    """Source 1: 从Crossref API获取摘要。"""
    url = f"https://api.crossref.org/works/{doi}"
    headers = {'User-Agent': f'PaperMatcher/1.0 (mailto:{YOUR_EMAIL})'}
    response = make_request(url, session, rate_limiter, headers=headers)
    
    if response and response.status_code == 200:
        try:
            data = response.json()
            abstract_html = data.get("message", {}).get("abstract")
            if abstract_html:
                clean_abstract = re.sub('<[^<]+?>', '', abstract_html).strip()
                if clean_abstract:
                    return clean_abstract
        except Exception:
            pass
    return None

def get_abstract_from_pubmed(doi, session, rate_limiter):
    """Source 2: 从PubMed API获取摘要。"""
    headers = {'User-Agent': f'PaperMatcher/1.0 (mailto:{YOUR_EMAIL})'}
    
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {'db': 'pubmed', 'term': doi, 'retmode': 'json'}
    search_response = make_request(search_url, session, rate_limiter, params=search_params, headers=headers)

    if not (search_response and search_response.status_code == 200):
        return None
    
    try:
        search_data = search_response.json()
        pmid_list = search_data.get("esearchresult", {}).get("idlist")
        if not pmid_list:
            return None
        pmid = pmid_list[0]
    except Exception:
        return None
        
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {'db': 'pubmed', 'id': pmid, 'rettype': 'abstract', 'retmode': 'xml'}
    fetch_response = make_request(fetch_url, session, rate_limiter, params=fetch_params, headers=headers)
    
    if not (fetch_response and fetch_response.status_code == 200):
        return None
        
    try:
        root = ET.fromstring(fetch_response.content)
        abstract_texts = [node.text for node in root.findall('.//AbstractText') if node.text]
        if abstract_texts:
            return " ".join(abstract_texts)
    except Exception:
        return None
    return None

def get_abstract_from_openalex(doi, session, rate_limiter):
    """Source 3: 从OpenAlex API获取摘要。"""
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    headers = {'User-Agent': f'PaperMatcher/1.0 (mailto:{YOUR_EMAIL})'}
    response = make_request(url, session, rate_limiter, headers=headers)
    
    if response and response.status_code == 200:
        try:
            data = response.json()
            inverted_abstract = data.get('abstract_inverted_index')
            abstract = deconstruct_abstract(inverted_abstract)
            return abstract if abstract else None
        except Exception:
            pass
    return None

def get_paper_abstract(doi, session, general_rate_limiter, pubmed_rate_limiter):
    """使用三级降级策略获取论文摘要 (Crossref -> PubMed -> OpenAlex)。"""
    if not doi or pd.isna(doi):
        return "DOI Invalid"
    
    # 清理DOI格式
    clean_doi = format_doi(doi)
    
    # 尝试 Crossref
    abstract = get_abstract_from_crossref(clean_doi, session, general_rate_limiter)
    if abstract:
        return abstract
    
    # 尝试 PubMed
    abstract = get_abstract_from_pubmed(clean_doi, session, pubmed_rate_limiter)
    if abstract:
        return abstract
    
    # 尝试 OpenAlex
    abstract = get_abstract_from_openalex(clean_doi, session, general_rate_limiter)
    if abstract:
        return abstract
    
    return "Abstract Not Available"

# --- 辅助函数 ---

def format_doi(doi_url):
    """将完整的DOI URL转换为纯DOI编号。"""
    if isinstance(doi_url, str):
        return doi_url.replace("https://doi.org/", "")
    return doi_url

def extract_id_from_url(url):
    """从 OpenAlex URL 中提取 ID 部分。"""
    if isinstance(url, str):
        return url.split('/')[-1]
    return None

def get_work_by_doi(doi, session, rate_limiter):
    """通过DOI从OpenAlex获取论文的详细信息，包含自动重试逻辑。"""
    if not doi or pd.isna(doi):
        return None
    
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    headers = {'User-Agent': f'PaperMatcher/1.0 (mailto:{YOUR_EMAIL})'}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            rate_limiter.wait() # 在请求前等待速率许可
            response = session.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # 增加对 429 (Too Many Requests) 错误的处理
            if e.response.status_code in [403, 429] and attempt < max_retries - 1:
                error_code = e.response.status_code
                wait_time = 30 * (attempt + 1)  # 遇到速率问题时，等待更长时间
                tqdm.write(f"\n  [!] 获取DOI {doi} 时收到 {error_code}。等待 {wait_time} 秒后重试... (尝试 {attempt + 2}/{max_retries})")
                time.sleep(wait_time)
                continue
            tqdm.write(f"  [!] 获取DOI {doi} 的数据时出现HTTP错误: {e}")
            return None
        except requests.exceptions.RequestException as e:
            tqdm.write(f"  [!] 获取DOI {doi} 的数据时出错: {e}")
            return None
    return None

def find_matching_works(journal_id, year, topic_id, session, rate_limiter):
    """在OpenAlex上查找符合条件的匹配论文，包含自动重试逻辑。"""
    if not all([journal_id, year, topic_id]):
        return []

    matches = []
    filters = [
        f"primary_location.source.id:{journal_id}",
        f"publication_year:{year}",
        f"concepts.id:{topic_id}",
        "is_retracted:false",
    ]
    filter_string = ",".join(filters)
    url_base = f"https://api.openalex.org/works?filter={filter_string}&per-page=200&cursor="
    cursor = '*'
    headers = {'User-Agent': f'PaperMatcher/1.0 (mailto:{YOUR_EMAIL})'}
    max_retries = 3
    
    while cursor:
        paginated_url = f"{url_base}{cursor}"
        
        for attempt in range(max_retries):
            try:
                rate_limiter.wait() # 在请求前等待速率许可
                response = session.get(paginated_url, headers=headers)
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])
                matches.extend(results)
                cursor = data.get('meta', {}).get('next_cursor', None)
                break
            except requests.exceptions.HTTPError as e:
                # 增加对 429 (Too Many Requests) 错误的处理
                if e.response.status_code in [403, 429] and attempt < max_retries - 1:
                    error_code = e.response.status_code
                    wait_time = 30 * (attempt + 1) # 遇到速率问题时，等待更长时间
                    tqdm.write(f"\n  [!] 查找匹配项时收到 {error_code}。等待 {wait_time} 秒后重试... (尝试 {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    tqdm.write(f"\n  [!] 查找匹配论文时出错: {e}")
                    return None
            except requests.exceptions.RequestException as e:
                tqdm.write(f"\n  [!] 查找匹配论文时出错: {e}")
                return None
        else:
            return None
            
    return matches

def select_matches(candidates, retracted_citation_count):
    """
    从候选论文中选出最多三篇不重复的匹配论文。
    优先级为：引用数最高 > 引用数最低 > 引用数最接近。
    """
    if not candidates:
        return {}

    for cand in candidates:
        cand['cited_by_count'] = cand.get('cited_by_count', 0)

    remaining_candidates = list(candidates)
    selected = {}
    
    if not remaining_candidates: return {}
    max_citation_paper = max(remaining_candidates, key=lambda x: x['cited_by_count'])
    selected["MaxCitation"] = max_citation_paper
    remaining_candidates = [p for p in remaining_candidates if p['id'] != max_citation_paper['id']]

    if not remaining_candidates: return selected
    min_citation_paper = min(remaining_candidates, key=lambda x: x['cited_by_count'])
    selected["MinCitation"] = min_citation_paper
    remaining_candidates = [p for p in remaining_candidates if p['id'] != min_citation_paper['id']]

    if not remaining_candidates: return selected
    closest_citation_paper = min(
        remaining_candidates,
        key=lambda x: abs(x['cited_by_count'] - retracted_citation_count)
    )
    selected["ClosestCitation"] = closest_citation_paper
    
    return selected

def process_paper(retracted_row, session, rate_limiter, pubmed_rate_limiter):
    """处理单篇撤稿论文：获取信息、查找匹配项并返回结果元组。"""
    record_id = retracted_row['Record ID']
    retracted_doi = retracted_row['OriginalPaperDOI']

    retracted_work = get_work_by_doi(retracted_doi, session, rate_limiter)

    if retracted_work is None:
        return ([], None)

    retracted_work_id_url = retracted_work.get('id')
    retracted_citation = retracted_work.get('cited_by_count', 0)
    retracted_title = retracted_work.get('title', 'N/A')
    pub_year = retracted_work.get('publication_year')
    journal = retracted_work.get('primary_location', {}).get('source', {})
    journal_id_url = journal.get('id') if journal else None
    journal_display_name = journal.get('display_name') if journal else "N/A"
    
    concepts = retracted_work.get('concepts', [])
    primary_topic = None
    if concepts:
        sorted_concepts = sorted(concepts, key=lambda c: c.get('score', 0), reverse=True)
        primary_topic = sorted_concepts[0]
        
    primary_topic_id_url = primary_topic.get('id') if primary_topic else None
    primary_topic_name = primary_topic.get('display_name') if primary_topic else "N/A"
    
    journal_id = extract_id_from_url(journal_id_url)
    primary_topic_id = extract_id_from_url(primary_topic_id_url)

    if not all([journal_id, pub_year, primary_topic_id]):
        queue_info = {'RetractedRecordID': record_id, 'RetractedDOI': retracted_doi, 'CandidateCount': 0}
        return ([], queue_info)

    candidates = find_matching_works(journal_id, pub_year, primary_topic_id, session, rate_limiter)

    if candidates is None:
        return ([], None)

    queue_info = {'RetractedRecordID': record_id, 'RetractedDOI': retracted_doi, 'CandidateCount': len(candidates)}

    if retracted_work_id_url:
        candidates = [p for p in candidates if p.get('id') != retracted_work_id_url]

    if not candidates:
        return ([], queue_info)

    selected_matches = select_matches(candidates, retracted_citation)
    
    if not selected_matches:
        return ([], queue_info)

    match_results = []
    
    # 获取撤稿论文的摘要
    retracted_abstract = get_paper_abstract(retracted_doi, session, rate_limiter, pubmed_rate_limiter)
    
    match_results.append({
        'RetractedRecordID': record_id, 'RetractedDOI': retracted_doi, 'RetractedPaperTitle': retracted_title, 
        'RetractedPaperCitationCount': retracted_citation, 'MatchType': 'Retracted', 'PaperDOI': format_doi(retracted_work.get('doi')),
        'PaperTitle': retracted_work.get('title'), 'PublicationYear': pub_year, 'Journal': journal_display_name,
        'CitationCount': retracted_citation, 'PrimaryTopic': primary_topic_name, 'Abstract': retracted_abstract
    })

    for match_type, paper in selected_matches.items():
        # 获取匹配论文的摘要
        paper_doi = format_doi(paper.get('doi'))
        paper_abstract = get_paper_abstract(paper_doi, session, rate_limiter, pubmed_rate_limiter)
        
        match_results.append({
            'RetractedRecordID': record_id, 'RetractedDOI': retracted_doi, 'RetractedPaperTitle': retracted_title,
            'RetractedPaperCitationCount': retracted_citation, 'MatchType': match_type, 'PaperDOI': paper_doi,
            'PaperTitle': paper.get('title'), 'PublicationYear': paper.get('publication_year'),
            'Journal': paper.get('primary_location', {}).get('source', {}).get('display_name', 'N/A'),
            'CitationCount': paper.get('cited_by_count', 0), 'PrimaryTopic': paper.get('concepts', [{}])[0].get('display_name', 'N/A'),
            'Abstract': paper_abstract
        })
    
    return (match_results, queue_info)

def save_dataframe(data_to_save, output_path):
    """将字典列表追加到CSV文件中。"""
    if not data_to_save:
        return
    output_df = pd.DataFrame(data_to_save)
    header = not os.path.exists(output_path)
    output_df.to_csv(output_path, mode='a', header=header, index=False, encoding='utf-8-sig')

# --- 主逻辑 ---

def main():
    """主函数，用于读取数据、处理论文并保存结果。"""
    print(f"--- 开始运行论文匹配脚本 ({NUM_WORKERS}个线程并发模式) ---")
    
    if YOUR_EMAIL == "your_email@example.com":
        print("\n\n致命错误: 请在脚本顶部将 'YOUR_EMAIL' 变量更改为您自己的真实邮箱地址。")
        return

    processed_ids = set()
    if os.path.exists(OUTPUT_CSV_PATH):
        print(f"检测到已存在的输出文件 '{OUTPUT_CSV_PATH}'，将从中加载进度。")
        try:
            try:
                # 关键修复：确保将 RetractedRecordID 读取为字符串，以避免类型不匹配
                processed_df = pd.read_csv(OUTPUT_CSV_PATH, dtype={'RetractedRecordID': str}, low_memory=False)
            except UnicodeDecodeError:
                # 如果UTF-8失败，则尝试latin1
                processed_df = pd.read_csv(OUTPUT_CSV_PATH, encoding='latin1', dtype={'RetractedRecordID': str}, low_memory=False)
            
            if 'RetractedRecordID' in processed_df.columns:
                processed_ids = set(processed_df['RetractedRecordID'].unique())
                print(f"已找到 {len(processed_ids)} 个已处理的记录。脚本将从断点处继续。")
        except pd.errors.EmptyDataError:
             print("警告: 输出文件为空，将从头开始。")
        except Exception as e:
            print(f"读取现有输出文件时出错: {e}。将重新开始处理。")

    try:
        try:
            retracted_df = pd.read_csv(INPUT_CSV_PATH, dtype={'Record ID': str, 'OriginalPaperDOI': str}, low_memory=False)
        except UnicodeDecodeError:
            print("  [!] 使用 UTF-8 编码读取失败。正在尝试 'latin1' 编码...")
            retracted_df = pd.read_csv(INPUT_CSV_PATH, encoding='latin1', dtype={'Record ID': str, 'OriginalPaperDOI': str}, low_memory=False)
        print(f"成功读取 {len(retracted_df)} 条撤稿记录。")
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    if processed_ids:
        original_count = len(retracted_df)
        retracted_df = retracted_df[~retracted_df['Record ID'].isin(processed_ids)]
        print(f"将跳过 {original_count - len(retracted_df)} 个已处理的记录，剩余 {len(retracted_df)} 个新记录需要处理。")
    
    if retracted_df.empty:
        print("所有记录均已处理完毕。")
        return

    results_to_save_buffer = []
    queue_lengths_to_save_buffer = []
    processed_papers_count = 0
    saving_lock = threading.Lock()
    session = requests.Session()
    rate_limiter = RateLimiter(API_REQUESTS_PER_SECOND)
    pubmed_rate_limiter = RateLimiter(PUBMED_API_REQUESTS_PER_SECOND)
    
    if PROXIES:
        session.proxies.update(PROXIES)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        rows_to_process = [row for _, row in retracted_df.iterrows()]
        futures = {executor.submit(process_paper, row, session, rate_limiter, pubmed_rate_limiter): row for row in rows_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理撤稿论文"):
            try:
                match_results, queue_info = future.result()
                
                with saving_lock:
                    if queue_info is not None:
                        queue_lengths_to_save_buffer.append(queue_info)
                        processed_papers_count += 1
                        if match_results:
                            results_to_save_buffer.extend(match_results)
                        
                        if processed_papers_count % SAVE_INTERVAL == 0:
                            tqdm.write(f"\n... 达到保存点，正在保存数据 ...")
                            save_dataframe(results_to_save_buffer, OUTPUT_CSV_PATH)
                            save_dataframe(queue_lengths_to_save_buffer, QUEUE_LENGTH_CSV_PATH)
                            results_to_save_buffer = []
                            queue_lengths_to_save_buffer = []
            except Exception as exc:
                row_data = futures[future]
                tqdm.write(f"\n  [!] 处理DOI {row_data.get('OriginalPaperDOI', 'N/A')} 时产生了一个未预料的错误: {exc}")

    if results_to_save_buffer or queue_lengths_to_save_buffer:
        print(f"\n... 循环结束，正在保存剩余的数据 ...")
        save_dataframe(results_to_save_buffer, OUTPUT_CSV_PATH)
        save_dataframe(queue_lengths_to_save_buffer, QUEUE_LENGTH_CSV_PATH)

    print(f"\n--- 处理完成 ---")
    print(f"匹配结果已保存到 '{OUTPUT_CSV_PATH}'。")
    print(f"匹配队列长度记录已保存到 '{QUEUE_LENGTH_CSV_PATH}'。")

if __name__ == "__main__":
    main()

