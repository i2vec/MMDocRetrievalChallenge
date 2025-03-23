import os
import json
import pandas as pd
import ray
from tqdm import tqdm
from qwen25vl_judger import ContentJudger

@ray.remote(num_gpus=1)
def process_chunk(chunk, queries, m2kr_passages, m2kr_query_img_root):
    # 每个worker内部实例化ContentJudger，确保占用对应GPU
    content_judger = ContentJudger()
    results = []
    for q in tqdm(chunk, desc="处理问题", total=len(chunk)):
        question_id = q['question_id']
        passage_ids = q['passage_ids']
        
        # 在查询数据中获取对应问题详细信息
        query_item = queries[queries['question_id'] == question_id]
        if query_item.empty:
            continue
        query_item = query_item.iloc[0]
        
        q_text = query_item.get("question", "").strip()
        if not q_text:
            q_text = "What does this image describe?"
        img_path = os.path.join(m2kr_query_img_root, query_item["img_path"])
        
        print(f"处理问题 {question_id}: {q_text}")
        
        yes_queue = []
        no_queue = []
        # 对每个passage进行内容相关性判断
        for pid in passage_ids:
            passage_row = m2kr_passages[m2kr_passages['passage_id'] == pid]
            if passage_row.empty:
                continue
            passage = passage_row.iloc[0]['passage_content']
            try:
                res, response = content_judger.judge_relevance(img_path, q_text, passage)
            except Exception as e:
                print(f"处理问题 {question_id} 的 passage {pid} 时出现异常: {e}")
                res = 'yes'

            if res == 'yes':
                yes_queue.append(pid)
            else:
                no_queue.append(pid)
        # 组合yes_queue和no_queue，并只保留前5个结果作为top5
        top5_passages = (yes_queue + no_queue)[:5]
        results.append({
            "question_id": question_id,
            "top5_passages": top5_passages,
            "yes_queue": yes_queue,
            "no_queue": no_queue
        })
    return results

if __name__ == "__main__":
    # 初始化Ray，并使用4个GPU并行worker
    ray.init()
    
    # 读取待处理问题文件
    with open("rerank_targets.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    # 配置M2KR相关路径
    m2kr_root = os.environ["M2KR_ROOT"]
    passages_path = os.path.join(m2kr_root, 'challenge_passage/train-00000-of-00001.parquet')
    m2kr_passages = pd.read_parquet(passages_path)
    
    queries_path = os.environ["M2KR_QUERY_PATH"]
    m2kr_query_img_root = os.environ["M2KR_QUERY_IMG_DIR"]
    queries = pd.read_parquet(queries_path)
    
    # 将问题划分为4个数据块，每个块由一个worker处理
    num_workers = 4
    chunk_size = len(questions) // num_workers
    chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        if i == num_workers - 1:
            end_idx = len(questions)
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append(questions[start_idx:end_idx])
    
    # 并行处理每个数据块
    tasks = [process_chunk.remote(chunk, queries, m2kr_passages, m2kr_query_img_root) for chunk in chunks]
    results_list = ray.get(tasks)
    results = []
    for res in results_list:
        results.extend(res)
    
    # 保存最终结果到输出文件
    output_file = os.environ["OUTPUT_FILE_M2KR_4"]
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=4, ensure_ascii=False)
    print(f"结果已保存至 {output_file}")
    
    ray.shutdown()
