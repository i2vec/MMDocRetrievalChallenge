import os
import json
import csv

with open(os.environ["OUTPUT_FILE_M2KR_4"], "r", encoding="utf-8") as f:
    mm2kr_data = json.load(f)

with open(os.environ["OUTPUT_FILE_MMDOCIR_4"], "r", encoding="utf-8") as f:
    mmdocir_data = json.load(f)

# 创建CSV文件并写入
with open(os.environ["OUTPUT_FILE_BEST"], "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])
    
    # 处理M2KR数据
    for item in mm2kr_data:
        question_id = item["question_id"]
        top5_passages = item["top5_passages"][:5]  # 只取前5个
        writer.writerow([str(question_id), json.dumps(top5_passages).replace("'", "\"")])
        
    # 处理MMDocIR数据  
    for item in mmdocir_data:
        question_id = item["question_id"]
        rerank_result = item["rerank_result"][:5]  # 只取前5个
        writer.writerow([str(question_id), json.dumps(rerank_result).replace("'", "\"")])
