import os
import pandas as pd
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from data.mmdocir_dataset import MMDocIRDataset
import ray
import uuid
import torch

# 初始化Ray
ray.init()

# 定义远程Worker类
@ray.remote(num_gpus=1)
class ModelWorker:
    def __init__(self, colqwen_path):
        # self.model = ColQwen2_5.from_pretrained(
        #     colqwen_path,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto"
        # ).eval()
        # self.processor = ColQwen2_5_Processor.from_pretrained(colqwen_path)
        self.model = ColQwen2.from_pretrained(
            colqwen_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(colqwen_path)
    
    def process_batch(self, batch_items, dataset_df, mmdocir_root):
        results = []
        for item in tqdm(batch_items, desc=f"处理批次 {uuid.uuid4().hex[:8]}", leave=False):
            doc_name = item["doc_name"]
            doc_pages = dataset_df.loc[dataset_df['doc_name'] == doc_name]
            image_list = []
            passage_id_list = []
            ocr_list = []
            for _, row in doc_pages.iterrows():
                image_path = os.path.join(mmdocir_root, row['image_path'])
                image_list.append(Image.open(image_path))
                passage_id_list.append(row['passage_id'])
                ocr_list.append(row['ocr_text'])
            
            # 如果数据为空，使用默认结果
            if len(passage_id_list) == 0:
                results.append({
                    "question_id": item["question_id"],
                    "text_to_text": {
                        "passage_ids": ["0"] * 10,
                        "scores": [0.0] * 10
                    }
                })
                continue
            
            try:
                # 获取问题和图片的embedding
                question_embedding = self.processor.process_queries([item['question']]).to(self.model.device)
                with torch.no_grad():
                    question_embedding = self.model(**question_embedding)
                image_embedding_list = []
                batch_size = 2
                for i in range(0, len(image_list), batch_size):
                    try:
                        def resize_image(image):
                            new_h, new_w = smart_resize(image.height, image.width)
                            image = image.resize((new_w, new_h))
                            return image
                        batch_images = [resize_image(image) for image in image_list[i:i + batch_size]]
                        image_input = self.processor.process_images(batch_images).to(self.model.device)
                        with torch.no_grad():
                            image_embeddings = self.model(**image_input)
                        image_embedding_list.extend([emb.squeeze(0) for emb in image_embeddings])
                    except Exception as e:
                        print(f"处理图片时出错: {str(e)}, question_id: {item['question_id']}")
                        continue

                scores_1 = self.processor.score_multi_vector(question_embedding, image_embedding_list)
                scores_1 = scores_1[0].cpu().numpy()
                # scores_2 = scores_2[0].cpu().numpy()
                # scores_5 = scores_5[0].cpu().numpy()
                
                # 处理三种不同的分数
                top10_indices_1 = np.argsort(scores_1).tolist()[::-1]
                # top10_indices_2 = np.argsort(scores_2).tolist()[::-1]
                # top10_indices_5 = np.argsort(scores_5).tolist()[::-1]
                
                # 确保每个列表至少有10个元素
                if len(top10_indices_1) < 10:
                    top10_indices_1 = top10_indices_1 + [top10_indices_1[0]] * (10 - len(top10_indices_1))
                # if len(top10_indices_2) < 10:
                    # top10_indices_2 = top10_indices_2 + [top10_indices_2[0]] * (10 - len(top10_indices_2))
                # if len(top10_indices_5) < 10:
                    # top10_indices_5 = top10_indices_5 + [top10_indices_5[0]] * (10 - len(top10_indices_5))
                
                # 获取前10个passage_ids和分数
                top10_passage_ids_1 = [passage_id_list[i] for i in top10_indices_1][:10]
                top10_scores_1 = [float(scores_1[i]) for i in top10_indices_1][:10]
                
                # top10_passage_ids_2 = [passage_id_list[i] for i in top10_indices_2][:10]
                # top10_scores_2 = [float(scores_2[i]) for i in top10_indices_2][:10]
                
                # top10_passage_ids_5 = [passage_id_list[i] for i in top10_indices_5][:10]
                # top10_scores_5 = [float(scores_5[i]) for i in top10_indices_5][:10]
                
                results.append({
                    "question_id": item["question_id"],
                    "text_to_image": {
                        "passage_ids_1": top10_passage_ids_1,
                        "scores_1": top10_scores_1,
                        # "passage_ids_2": top10_passage_ids_2,
                        # "scores_2": top10_scores_2,
                        # "passage_ids_5": top10_passage_ids_5,
                        # "scores_5": top10_scores_5
                    }
                })
                
            except Exception as e:
                print(f"模型推理时出错: {str(e)}, question_id: {item['question_id']}")
                results.append({
                    "question_id": item["question_id"],
                    "text_to_text": {
                        "passage_ids": ["0"] * 10,
                        "scores": [0.0] * 10
                    }
                })
        return results

def main(debug=False):
    # 模型和数据路径
    colqwen_path = os.environ["COLQWEN2_7B_PATH"]
    mmdocir_dataset = MMDocIRDataset()

    if debug:
        print(f"debug模式，只处理前10条数据")
        mmdocir_dataset = mmdocir_dataset[:10]
    # 创建4个worker
    num_workers = 4
    workers = [ModelWorker.remote(colqwen_path) for _ in range(num_workers)]
    
    # 按文档名称排序,确保相似的文档分散到不同worker
    mmdocir_dataset.sort(key=lambda x: x["doc_name"])
    
    # 准备批次数据,交错分配确保每个worker获得均匀的文档分布
    batches = [[] for _ in range(num_workers)]
    for i, item in enumerate(mmdocir_dataset):
        worker_idx = i % num_workers
        batches[worker_idx].append(item)
    
    # 并行处理
    print("开始并行处理数据...")
    results_refs = [
        workers[i].process_batch.remote(batches[i])
        for i in range(num_workers)
    ]
    
    # 获取结果
    final_results = []
    for result_ref in tqdm(results_refs, desc="等待批次完成"):
        batch_results = ray.get(result_ref)
        final_results.extend(batch_results)
        
    # 按question_id排序
    final_results.sort(key=lambda x: x["question_id"])
    
    # 保存结果
    with open(os.environ["OUTPUT_FILE_MMDOCIR_1"], "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print("处理完成！")

if __name__ == "__main__":
    main()