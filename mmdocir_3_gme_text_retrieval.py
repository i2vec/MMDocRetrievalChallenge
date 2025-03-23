import os
import json
from tqdm import tqdm
import numpy as np
from models.gme import GmeQwen2VL
import chromadb
from data.mmdocir_dataset import MMDocIRDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(debug=os.environ["DEBUG"]=="true"):
    # 初始化模型和数据
    mmdocir_dataset = MMDocIRDataset()
    chromadb_path = os.environ["MMDOCIR_PASSAGES_TEXT_EMBEDDING_CHROMA_PATH"]
    gme_model = GmeQwen2VL(
        model_path=os.environ["GME_PATH"]
    )
    client = chromadb.PersistentClient(path=chromadb_path)
    
    
    if debug:
        print("debug模式,只处理前20条数据")
        data_items = data_items[:10]
        
    results = []
    for item in tqdm(mmdocir_dataset):
        question = item["question"]
        question_id = item["question_id"]
        doc_name = item["doc_name"]
        
        if len(doc_name) < 4:
            doc_name = f"xxxx{doc_name}"
        elif len(doc_name) > 60:
            doc_name = f"xxxx{doc_name[:50]}"
            
        try:
            # 获取问题的embedding
            question_embedding = gme_model.get_text_embeddings([question]).numpy()  # shape: (1, d)
            
            # 从collection中检索相似文本，并获取文档的embedding以便用点积计算score（参照file_context_0）
            collection = client.get_collection(doc_name)
            query_results = collection.query(
                query_embeddings=question_embedding,
                n_results=30,
                include=["metadatas", "embeddings"]
            )
            
            passage_scores = []
            
            if len(query_results["metadatas"][0]) > 0:
                # 收集passage_id和点积分数（score）
                for metadata, doc_emb in zip(query_results["metadatas"][0], query_results["embeddings"][0]):
                    pid = metadata["passage_id"]
                    # 计算点积分数，类似于: scores = (batch_queries * batch_doc_embeddings).sum(-1)
                    score = (question_embedding * np.array(doc_emb)).sum()
                    passage_scores.append((pid, score))
                
                # 按score从大到小排序（高score表示文本更相似）
                passage_scores.sort(key=lambda x: x[1], reverse=True)
                # 去重：只保留重复项中靠前的（即最高score的那一项）
                unique_passage_scores = []
                seen_pids = set()
                for pid, score in passage_scores:
                    if pid not in seen_pids:
                        unique_passage_scores.append((pid, score))
                        seen_pids.add(pid)
                passage_scores = unique_passage_scores
            
            # # 如果结果少于30个,用默认值填充（默认score设为 -inf 表示最小相似度）
            # while len(passage_scores) < 30:
            #     passage_scores.append(("0", float('-inf')))
            
            # 分离排序后的passage_ids和scores
            passage_ids = [p[0] for p in passage_scores[:30]]
            scores = [p[1] for p in passage_scores[:30]]
                
            results.append({
                "question_id": question_id,
                "text_to_text": {
                    "passage_ids": passage_ids,
                    "scores": scores
                }
            })
            
        except Exception as e:
            print(f"处理问题时出错: {str(e)}, question_id: {question_id}")
            results.append({
                "question_id": question_id,
                "text_to_text": {
                    "passage_ids": ["0"] * 30,
                    "scores": [float('-inf')] * 30
                }
            })
    
    # 按question_id排序
    results.sort(key=lambda x: x["question_id"])
    
    # 保存结果（文件名已更新以反映使用score计算）
    with open("", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("处理完成！")

if __name__ == "__main__":
    main()
