import os
import json
import csv

# 读取M2KR结果
with open(os.environ["OUTPUT_FILE_M2KR_3"], "r", encoding="utf-8") as f:
    m2kr_data = json.load(f)

# 读取MMDocIR结果
with open(os.environ["OUTPUT_FILE_MMDOCIR_2_JSON"], "r", encoding="utf-8") as f:
    mmdocir_data_layout = json.load(f)

with open(os.environ["OUTPUT_FILE_MMDOCIR_3"], "r", encoding="utf-8") as f:
    mmdocir_data_text = json.load(f)

# 合并 layout 和 text 结果
mmdoc_result = {}

# 处理layout数据格式
layout_formatted = {}
for qid, data in mmdocir_data_layout.items():
    passage_ids = json.loads(data["passage_id"])
    scores = data["scores"]
    layout_formatted[qid] = dict(zip(passage_ids, scores))

# 处理text数据格式  
text_formatted = {}
for item in mmdocir_data_text:
    qid = item["question_id"]
    text_data = item["text_to_text"]
    passage_ids = text_data["passage_ids"]
    scores = text_data["scores"]
    text_formatted[qid] = dict(zip(passage_ids, scores))

# 合并所有问题ID
all_questions = set(layout_formatted.keys()).union(text_formatted.keys())
for question in sorted(all_questions, key=int):
    layout_scores = layout_formatted.get(question, {})
    text_scores = text_formatted.get(question, {})
    combined = {}
    
    # 合并 layout 和 text 的 passage id
    all_pids = set(layout_scores.keys()).union(text_scores.keys())
    for pid in all_pids:
        score_layout = layout_scores.get(pid, float('-inf')) * 0.6  # layout权重0.6
        score_text = text_scores.get(pid, float('-inf')) * 0.4      # text权重0.4
        combined[pid] = score_layout + score_text
    # 按加权后的分数降序排序
    sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    final_pids = [pid for pid, score in sorted_combined if score >= float('-inf')]
    # 如果passage数量不足5个,重复最高分passage直到补齐
    if final_pids and len(final_pids) < 5:
        while len(final_pids) < 5:
            final_pids.append(final_pids[0])
            
    mmdoc_result[question] = final_pids[:5]



# 创建CSV文件并写入
with open(os.environ["OUTPUT_FILE_UNIFIED"], "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "passage_id"])
    
    # 处理M2KR数据
    for item in m2kr_data:
        question_id = item["question_id"]
        top5_passages = item["passage_id"][:5]  # 只取前5个
        writer.writerow([str(question_id), json.dumps(top5_passages).replace("'", "\"")])
    
    # 处理MMDocIR数据
    for question_id in sorted(mmdoc_result.keys(), key=int):
        top5_passages = mmdoc_result[question_id][:5]  # 只取前5个
        writer.writerow([str(question_id), json.dumps(top5_passages).replace("'", "\"")])
