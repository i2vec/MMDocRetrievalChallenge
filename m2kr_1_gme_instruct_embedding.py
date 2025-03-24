import os
import chromadb
from models.gme import GmeQwen2VL
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
import ray
from data.m2kr_dataset import M2KRPassagesDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, gme_path):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.gme_model = GmeQwen2VL(model_path=gme_path)
    
    def process_batch(self, batch_items):
        texts = [row['passage_content'] for row in batch_items]
        passage_ids = [row['passage_id'] for row in batch_items]
        
        try:
            with torch.no_grad():
                document_embeddings = self.gme_model.get_text_embeddings(
                    texts=texts, 
                    instruction='Find an image that matches the given text.'
                )
            return document_embeddings.tolist(), passage_ids
            
        except Exception as e:
            print(f"处理批次时发生错误: {str(e)}")
            return [], []

def main(
    gme_path: str = os.environ["GME_PATH"],
    chroma_path: str = os.environ["M2KR_PASSAGES_TEXT_EMBEDDING_CHROMA_PATH"],
    collection_name: str = os.environ['M2KR_PASSAGES_TEXT_EMBEDDING_CHROMA_COLLECTION_NAME'], 
    num_workers: int = torch.cuda.device_count(),
    debug: bool = os.environ["DEBUG"] == "true"
):
    ray.init()
    m2kr_passages_dataset = M2KRPassagesDataset()
    m2kr_passages = m2kr_passages_dataset

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)
    
    if debug:
        print("Debug mode, only process 10 passages")
        m2kr_passages = m2kr_passages[:10]
    
    workers = [ModelWorker.remote(gme_path) for _ in range(num_workers)]
    
    batch_size = len(m2kr_passages) // num_workers
    batches = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < num_workers - 1 else len(m2kr_passages)
        batches.append(m2kr_passages[start_idx:end_idx])
    
    print("开始并行处理数据...")
    results_refs = [
        workers[i].process_batch.remote(batches[i])
        for i in range(num_workers)
    ]
    
    for result_ref in tqdm(results_refs, desc="等待批次完成"):
        document_embeddings, passage_ids = ray.get(result_ref)
        collection.add(
            ids=passage_ids,
            embeddings=document_embeddings
        )
    
    print("处理完成！")

if __name__ == "__main__":
    main()