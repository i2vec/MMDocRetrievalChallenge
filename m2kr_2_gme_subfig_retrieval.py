from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
import chromadb
import json
import ray
import uuid
from collections import defaultdict

from models.gme import GmeQwen2VL

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
            device="cuda"
        )
        self.client = chromadb.PersistentClient(path="./chroma_db_2")
        self.collection = self.client.get_or_create_collection("m2kr_image")

    def process_batch(self, batch_items, m2kr_query_img_root, topk):
        results = []

        for query in tqdm(batch_items, desc=f"处理批次 {uuid.uuid4().hex[:8]}", leave=False, total=len(batch_items)):
            query_type = "fused"
            question_image_path = os.path.join(m2kr_query_img_root, query['img_path'])
            question_image = Image.open(question_image_path).convert("RGB")
            if query['question']:
                question_text = query['question']
                embedding = self.gme.get_fused_embeddings(
                    texts=[question_text],
                    images=[question_image],
                    # instruction="Find a text description that matches the given question and the image."
                )
                text_embedding = None
                # text_embedding = self.gme.get_text_embeddings(
                #     texts=[question_text + f"\nImage description: {query['lagend']}"],
                #     # instruction="Find a text description that matches the given question and the image description."
                # )
            else:
                query_type = "single"

                instruction = "Find a image that matches the given image."
                # v0
                # embedding = self.gme.get_image_embeddings(images=[question_image], instruction=instruction)
                # v1
                embedding = self.gme.get_fused_embeddings(images=[question_image], texts=[instruction])
                text_embedding = None
                # text_embedding = self.gme.get_text_embeddings(
                #     texts=[f"Image description: {query['lagend']}"],
                #     # instruction="Find a text description that matches the given image description."
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


# 将数据分成多个批次
def split_into_batches(df, num_workers):
    batch_size = len(df) // num_workers
    if len(df) % num_workers != 0:
        batch_size += 1

    batches = []
    for i in range(0, len(df), batch_size):
        batches.append(df.iloc[i:i + batch_size].to_dict('records'))

    return batches


# 主处理流程
def main(
        m2kr_query_img_root: str = os.environ.get("M2KR_QUERY_IMG_DIR"),
        queries_path: str = os.environ.get("M2KR_QUERY_PATH"),
        model_path: str = os.environ.get("GME_PATH"),
        num_workers: int = 4,
        topk: int = 100,
        output_dir: str = "./outputs",
        debug:bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    m2kr_queries = pd.read_parquet(queries_path)
    debug = (os.environ["DEBUG"] == "true")
    if debug:
        # m2kr_queries = m2kr_queries.sample(n=20, random_state=42)
        m2kr_queries = m2kr_queries[:10]

    # 创建多个worker
    workers = [ModelWorker.remote(model_path) for _ in range(num_workers)]

    # 分割数据
    batches = split_into_batches(m2kr_queries, num_workers)

    # 并行处理
    futures = [workers[i % num_workers].process_batch.remote(batch, m2kr_query_img_root, topk)
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
    with open(os.environ.get("OUTPUT_FILE_M2KR_2"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('retrieval of m2kr gme subfig finished.')


if __name__ == "__main__":
    main(debug=False)
