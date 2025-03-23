export DATA_ROOT="/mnt/vepfs/fs_ckps/xumj/data/wwwMrag"

export MMDOCIR_ROOT="${DATA_ROOT}/MMDocIR-Challenge" 
export M2KR_ROOT="${DATA_ROOT}/M2KR-Challenge"


export MMDOCIR_QUESTION_PATH="${MMDOCIR_ROOT}/MMDocIR_gt_remove.jsonl"
export MMDOCIR_PASSAGE_PATH="${MMDOCIR_ROOT}/MMDocIR_doc_passages.parquet"

export M2KR_QUERY_IMG_DIR="${M2KR_ROOT}/query_images"
export M2KR_QUERY_PATH="${M2KR_ROOT}/challenge_data/train-00000-of-00001.parquet"
export M2KR_PASSAGES_PATH="${M2KR_ROOT}/challenge_passage/train-00000-of-00001.parquet"
export M2KR_PASSAGES_IMG_DIR="${M2KR_ROOT}/Challenge"

# Test MMDocIR Dataset
echo "Test MMDocIR Dataset..."
python data/mmdocir_dataset.py

# Test M2KR Dataset
echo "Test M2KR Dataset..."
python data/m2kr_dataset.py


# Debug Mode
export DEBUG=true

# M2KR task
## 1. Execute text embedding
export GME_PATH="/mnt/vepfs/fs_ckps/xumj/models/Mrag/gme-Qwen2-VL-7B-Instruct"
export M2KR_PASSAGES_TEXT_EMBEDDING_CHROMA_PATH="./chroma/m2kr_text"
export M2KR_PASSAGES_TEXT_EMBEDDING_CHROMA_COLLECTION_NAME="m2kr_gme_instruct_text"
export OUTPUT_FILE_M2KR_1="./outputs/gme_image_to_text_retrieval.json" # The output file
python m2kr_1_gme_instruct_embedding.py
python m2kr_1_gme_instruct_retrieval.py


## 2. Execute layout image embedding and retrieval
export OUTPUT_FILE_M2KR_2="" # The output file


## 3. Merge the results of 1 and 2, and get the top10 result of each question
export OUTPUT_FILE_M2KR_3="" # The output file
# <complelte and run the code here>

## 4. Rerank
export OUTPUT_FILE_M2KR_4="./output/vlrerank_gme_image_to_text_retrieval.json" # The output file
python M2KRContentJudger/run_qwen25vl_judger.py


# MMDocIR task
## 1. execute colqwen
export COLQWEN2_7B_PATH="/mnt/vepfs/fs_ckps/xumj/models/Mrag/colqwen2-7b-v1.0"
export OUTPUT_FILE_MMDOCIR_1="./outputs/mmdocir_colqwen2_7b_retrieval_top10.json" # The output file
python mmdocir_1_colqwen.py

## 2. execute gme layout retrieval
export OUTPUT_FILE_MMDOCIR_2"" # The output file
# <complelte and run the code here>

## 3. execute gme text retrieval
export MMDOCIR_PASSAGES_TEXT_EMBEDDING_CHROMA_PATH="./chroma/mmdocir_text"
export OUTPUT_FILE_MMDOCIR_3="./outputs/mmdocir_gme_text_retrieval_500_20_top30_scores.json" # The output file
python mmdocir_3_gme_text_embed.py
python mmdocir_3_gme_text_retrieval.py


## 4. rerank the 1
# <complelte and run the code here>
export OUTPUT_FILE_MMDOCIR_4="./outputs/vlrerank_colqwen2_7b.json" # The output file
python MMDocIRContentJudger/run_qwen25vl_judger.py


## 5. merge the results 2 and 3, and get the top10 result of each question
# <complelte and run the code here>
export OUTPUT_FILE_MMDOCIR_5="" # The output file

# Get The Final Submission File
## 1. Best
export OUTPUT_FILE_BEST="" # The output file
# <complelte and run the code here>

## 2. Single Best
export OUTPUT_FILE_SINGLE_BEST="" # The output file
# <complelte and run the code here>