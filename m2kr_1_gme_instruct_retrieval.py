from tqdm import tqdm
from PIL import Image
import os   
import pandas as pd
from models.gme import GmeQwen2VL
import chromadb
import json
import ray
import uuid
from collections import defaultdict
import torch
from data.m2kr_dataset import M2KRQuestionDataset
# 初始化Ray
ray.init()

# 定义远程Worker类
@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, model_path):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.gme = GmeQwen2VL(
            model_name="gme-Qwen2-VL-7B-Instruct",
            model_path=model_path,
            device="cuda:0"
        )
        self.client = chromadb.PersistentClient(path=os.environ["M2KR_PASSAGES_TEXT_EMBEDDING_CHROMA_PATH"])
        self.collection = self.client.get_or_create_collection(os.environ["M2KR_PASSAGES_TEXT_EMBEDDING_CHROMA_COLLECTION_NAME"])
    
    def process_batch(self, batch_items, topk):
        results = []

        for query in tqdm(batch_items, desc=f"处理批次 {uuid.uuid4().hex[:8]}", leave=False, total=len(batch_items)):
            query_type = "fused"
            question_image_path = query['image_path']
            question_image = Image.open(question_image_path).convert("RGB")
            if query['question']:
                question_text = query['question']
                embedding = self.gme.get_fused_embeddings(
                    texts=[question_text], 
                    images=[question_image], 
                    # instruction="Find a text description that matches the given question and the image."
                )
                # text_embedding = self.gme.get_text_embeddings(
                #     texts=[f"Image description: {query['lagend']}\n" + question_text], 
                #     instruction='Find an image that matches the given text.'
                #     # instruction="Find a text description that matches the given question and the image description."
                # )
            else: 
                query_type = "single"
                
                instruction = "Find a text description that matches the given image."
                # v0
                # embedding = self.gme.get_image_embeddings(images=[question_image], instruction=instruction)
                # v1
                embedding = self.gme.get_fused_embeddings(images=[question_image], texts=[instruction])

                # text_embedding = self.gme.get_text_embeddings(
                #     texts=[f"Image description: {query['lagend']}"], 
                #     # instruction="Find a text description that matches the given image description."
                #     instruction='Find an image that matches the given text.'
                # )
            
            # 计算top10
            query_results = self.collection.query(
                query_embeddings=embedding.cpu().numpy().tolist(),
                n_results=topk,
                include=['distances']
            )
            # query_results_legend = self.collection.query(
            #     query_embeddings=text_embedding.cpu().numpy().tolist(),
            #     n_results=topk,
            #     include=['distances']
            # )
            
            results.append({
                "question_id": query['question_id'],
                "question_type": query_type,
                "fused_to_text": {
                    "passage_ids": query_results['ids'][0],
                    "scores": query_results['distances'][0],
                },
                # "text_to_text": {
                #     "passage_ids": query_results_legend['ids'][0],
                #     "scores": query_results_legend['distances'][0],
                # }
            })
        
        return results

# 主处理流程
def main(
    model_path: str = os.environ["GME_PATH"],
    num_workers: int = torch.cuda.device_count(),
    topk: int = 10,
    output_path: str=os.environ["OUTPUT_FILE_M2KR_1"], 
    debug=os.environ["DEBUG"] == "true"
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    m2kr_dataset = M2KRQuestionDataset()
    
    if debug:
        m2kr_dataset = m2kr_dataset[:10]
    print(len(m2kr_dataset))
    print(debug)

    # 创建多个worker
    workers = [ModelWorker.remote(model_path) for _ in range(num_workers)]
    
        # 准备批次数据
    batch_size = len(m2kr_dataset) // num_workers
    batches = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < num_workers - 1 else len(m2kr_dataset)
        batches.append(m2kr_dataset[start_idx:end_idx])
    
    # 并行处理
    futures = [workers[i % num_workers].process_batch.remote(batch, topk) 
              for i, batch in enumerate(batches)]
    
    # 收集结果
    all_results = []
    for future in tqdm(futures, desc="收集结果"):
        batch_results = ray.get(future)
        all_results.extend(batch_results)
    
    # 按question_id排序
    all_results.sort(key=lambda x: int(x["question_id"]))
    
    # 按照question_id去重并整理结果
    results = []
    seen = set()
    
    for result in all_results:
        results.append(result)
        seen.add(result['question_id'])
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
