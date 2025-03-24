import os
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
import ray
from MMDocIRContentJudger.qwen25vl_judger import ContentJudger

@ray.remote(num_gpus=1)
def process_chunk(chunk, question_items, passage_df, mmdocir_root):
    """处理一个数据块的所有问题"""
    judger = ContentJudger()
    results = []
    for data in tqdm(chunk, total=len(chunk)):
        passage_ids = data['text_to_image']['passage_ids_1']
        question_id = data['question_id']
        
        # 找到对应的问题信息
        question_info = next((item for item in question_items if item['question_id'] == question_id), None)
        if question_info:
            question = question_info['question']
            doc_name = question_info['doc_name']
            
            # 本地处理，不再创建远程任务
            target_df = passage_df[passage_df['doc_name'] == doc_name]
            yes_queue = []
            no_queue = []
            
            for passage_id in passage_ids:
                target_page = target_df[target_df['passage_id']==passage_id].iloc[0]
                ocr_text = target_page['ocr_text']
                image_path = target_page['image_path']
                image_path = os.path.join(mmdocir_root, image_path)
                image = Image.open(image_path)
                result, response = judger.judge_relevance(image, question, ocr_text)
                if result == 'yes':
                    yes_queue.append(passage_id)
                else:
                    no_queue.append(passage_id)
            
            final_queue = yes_queue + no_queue
            ans_item = {
                "question_id": question_id, 
                "rerank_result": final_queue, 
                "yes_queue": yes_queue,
                "no_queue": no_queue
            }
            # print(ans_item['question_id'], passage_ids, "yes: ", yes_queue, "no: ", no_queue)
            results.append(ans_item)
    
    return results

def main(mmdocir_path=os.environ["OUTPUT_FILE_MMDOCIR_1"], 
         mmdocir_root=os.environ["MMDOCIR_ROOT"],
         num_workers=4,
         debug=False):  # 修改为worker数量参数
    """主函数，处理所有问题并保存结果"""
    # 初始化Ray
    ray.init()
    
    # 加载数据
    passage_path = os.path.join(mmdocir_root, 'MMDocIR_doc_passages.parquet')
    passage_df = pd.read_parquet(passage_path)
    
    # 加载问题
    question_items = []
    for line in open(os.path.join(mmdocir_root, "MMDocIR_gt_remove.jsonl"), 'r', encoding="utf-8"):
        question_items.append(json.loads(line.strip()))
    
    # 加载检索结果
    with open(mmdocir_path, "r") as f:
        mmdocir_answers = json.load(f)
    
    if debug:
        mmdocir_answers = mmdocir_answers[:10]
    # 将数据分为4份
    chunk_size = len(mmdocir_answers) // num_workers
    chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_workers - 1 else len(mmdocir_answers)
        chunks.append(mmdocir_answers[start_idx:end_idx])
    
    # 并行处理每个数据块
    chunk_tasks = []
    for chunk in chunks:
        task = process_chunk.remote(chunk, question_items, passage_df, mmdocir_root)
        chunk_tasks.append(task)
    
    # 获取所有任务结果并合并
    chunk_results = ray.get(chunk_tasks)
    vl_rerank_results = []
    for results in chunk_results:
        vl_rerank_results.extend(results)
    
    # 按照question_id排序
    vl_rerank_results.sort(key=lambda x: x['question_id'])
    
    # 保存结果
    output_path = os.environ["OUTPUT_FILE_MMDOCIR_4"]
    with open(output_path, "w") as f:
        json.dump(vl_rerank_results, f, indent=4, ensure_ascii=False)
    
    print(f"结果已保存到 {output_path}")
    
    # 关闭Ray
    ray.shutdown()

if __name__ == "__main__":
    main(debug=False)