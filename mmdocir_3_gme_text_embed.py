from tqdm import tqdm
from models.gme import GmeQwen2VL
from data.mmdocir_dataset import MMDocIRDataset
import os
import chromadb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

gme_model = GmeQwen2VL(
    model_path=os.environ["GME_PATH"]
)

mmdocir_dataset = MMDocIRDataset()

client = chromadb.PersistentClient(
    path=os.environ["MMDOCIR_PASSAGES_TEXT_EMBEDDING_CHROMA_PATH"]
)

from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return gme_model.get_text_embeddings(input).numpy()

embedding_model = MyEmbeddingFunction()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)


for data_item in tqdm(mmdocir_dataset):
    question = data_item["question"]
    question_id = data_item["question_id"]
    targets = data_item["targets"]
    doc_name = data_item["doc_name"]
    if len(doc_name) < 4:
        doc_name = f"xxxx{doc_name}"
    elif len(doc_name) >60:
        doc_name = f"xxxx{doc_name[:50]}"
    if doc_name in client.list_collections():
        continue

    collection = client.get_or_create_collection(
        doc_name, 
        embedding_function=embedding_model
    )

    for target in targets:
        ocr_text = target["ocr_text"]
        vlm_text = target["vlm_text"]
        passage_id = target["passage_id"]
        chunks = text_splitter.split_text(ocr_text)
        if len(chunks) == 0:
            continue
        collection.add(
            ids=[f"{passage_id}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"question_id": question_id, "passage_id": passage_id}] * len(chunks)
        )




