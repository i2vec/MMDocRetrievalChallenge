export DATA_ROOT="/the/root/of/the/data/data/wwwMrag"

export MMDOCIR_ROOT="${DATA_ROOT}/MMDocIR-Challenge" 
export M2KR_ROOT="${DATA_ROOT}/M2KR-Challenge"


export MMDOCIR_QUESTION_PATH="${MMDOCIR_ROOT}/MMDocIR_gt_remove.jsonl"
export MMDOCIR_PASSAGE_PATH="${MMDOCIR_ROOT}/MMDocIR_doc_passages.parquet"

export M2KR_QUERY_IMG_DIR="${M2KR_ROOT}/query_images"
export M2KR_QUERY_PATH="${M2KR_ROOT}/challenge_data/train-00000-of-00001.parquet"
export M2KR_PASSAGES_PATH="${M2KR_ROOT}/challenge_passage/train-00000-of-00001.parquet"
export M2KR_PASSAGES_IMG_DIR="${M2KR_ROOT}/Challenge"

# 创建outputs目录
if [ ! -d "./outputs" ]; then
    mkdir -p ./outputs
fi

# 检查 Python 环境
echo "当前使用的 Python 路径:"
which python

# Test MMDocIR Dataset
echo "Test MMDocIR Dataset..."
python data/mmdocir_dataset.py

# Test M2KR Dataset
echo "Test M2KR Dataset..."
python data/m2kr_dataset.py


# Debug Mode
export DEBUG=true

# M2KR task
# 1. Execute text embedding
export GME_PATH="/the/root/of/the/data/models/Mrag/gme-Qwen2-VL-7B-Instruct"
export QWEN_25_VL_72B_AWQ_PATH="/the/root/of/the/data/llms/Qwen2.5-VL-72B-Instruct-AWQ"
export M2KR_PASSAGES_TEXT_EMBEDDING_CHROMA_PATH="./chroma/m2kr_text"
export M2KR_PASSAGES_TEXT_EMBEDDING_CHROMA_COLLECTION_NAME="m2kr_gme_instruct_text"
export OUTPUT_FILE_M2KR_1="./outputs/gme_image_to_text_retrieval.json" # The output file
echo "Execute text embedding..."
python m2kr_1_gme_instruct_embedding.py
echo "Execute retrieval..."
python m2kr_1_gme_instruct_retrieval.py


# 2. Execute layout image embedding and retrieval
export OUTPUT_FILE_M2KR_2="./outputs/m2kr_subfig_match.json" # The output file
echo "Execute layout image embedding..."
python m2kr_2_gme_instruct_subfig_embedding.py
echo "Execute retrieval..."
python m2kr_2_gme_subfig_retrieval.py
## 3. Merge the results of 1 and 2, and get the top10 result of each question
export OUTPUT_FILE_M2KR_3="./outputs/m2kr_subfig_merge.json" # The output file
echo "Merge the results of 1 and 2..."
python m2kr_2_merge_m2kr_subfig_fused.py
## 4. Rerank
export OUTPUT_FILE_M2KR_4="./outputs/vlrerank_gme_image_to_text_retrieval.json" # The output file
echo "Rerank..."
python m2kr_run_qwen25vl_judger.py


# MMDocIR task
## 1. execute colqwen
export COLQWEN2_7B_PATH="/the/root/of/the/data/models/Mrag/colqwen2-7b-v1.0"
export OUTPUT_FILE_MMDOCIR_1="./outputs/mmdocir_colqwen2_7b_retrieval_top10.json" # The output file
echo "Execute colqwen retrieval..."
python mmdocir_1_colqwen.py

## 2. execute gme layout retrieval
export OUTPUT_FILE_MMDOCIR_2="./outputs/mmdocir_gme_layout.csv" # The output file
export OUTPUT_FILE_MMDOCIR_2_JSON="./outputs/mmdocir_gme_layout.json"
echo "Execute gme layout embedding..."
python mmdocir_2_gme_layout_embed.py
echo "Execute gme layout retrieval..."
python mmdocir_2_gme_layout_retrieval.py

## 3. execute gme text retrieval
export MMDOCIR_PASSAGES_TEXT_EMBEDDING_CHROMA_PATH="./chroma/mmdocir_text"
export OUTPUT_FILE_MMDOCIR_3="./outputs/mmdocir_gme_text_retrieval_500_20_top30_scores.json" # The output file
echo "Execute gme text embedding..."
python mmdocir_3_gme_text_embed.py
echo "Execute gme text retrieval..."
python mmdocir_3_gme_text_retrieval.py


## 4. rerank the 1
export OUTPUT_FILE_MMDOCIR_4="./outputs/vlrerank_colqwen2_7b.json" # The output file
echo "Execute rerank..."
python mmdocir_run_qwen25vl_judger.py


# Get The Final Submission File
## 1. Best
export OUTPUT_FILE_BEST="./outputs/submission_best.csv" # The output file
echo "Get the best result..."
python submission_best.py

## 2. Single Best
export OUTPUT_FILE_UNIFIED="./outputs/submission_unified.csv" # The output file
echo "Get the single best result..."
python submission_unified.py