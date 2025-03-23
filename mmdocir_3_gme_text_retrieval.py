import os
import json
from tqdm import tqdm
import numpy as np
from models.gme import GmeQwen2VL
import chromadb
from data.mmdocir_dataset import MMDocIRDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(debug=os.environ["DEBUG"]=="true"):
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
            question_embedding = gme_model.get_text_embeddings([question]).numpy()  # shape: (1, d)
            
            collection = client.get_collection(doc_name)
            query_results = collection.query(
                query_embeddings=question_embedding,
                n_results=30,
                include=["metadatas", "embeddings"]
            )
            
            passage_scores = []
            
            if len(query_results["metadatas"][0]) > 0:
                for metadata, doc_emb in zip(query_results["metadatas"][0], query_results["embeddings"][0]):
                    pid = metadata["passage_id"]
                    score = (question_embedding * np.array(doc_emb)).sum()
                    passage_scores.append((pid, score))
                
                passage_scores.sort(key=lambda x: x[1], reverse=True)
                unique_passage_scores = []
                seen_pids = set()
                for pid, score in passage_scores:
                    if pid not in seen_pids:
                        unique_passage_scores.append((pid, score))
                        seen_pids.add(pid)
                passage_scores = unique_passage_scores
            
            
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
    

    results.sort(key=lambda x: x["question_id"])
    
    with open("", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("处理完成！")

if __name__ == "__main__":
    main()
