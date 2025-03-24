import io
import json
import os.path
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm
from transformers.utils import is_flash_attn_2_available

from torch.utils.data import DataLoader, Dataset

import utils.crop_papers
from models.gme import GmeQwen2VL

save_csv_file = os.environ.get("OUTPUT_FILE_MMDOCIR_2")
save_json_file = os.environ.get("OUTPUT_FILE_MMDOCIR_2_JSON")


def get_instruction(query):
    default = "image"
    if "figure" in query.lower():
        default = "figure"
    if "table" in query.lower():
        default = "table"
    if "page" in query.lower():
        default = "page"
    template = f"Find an {default} that can solve the given question."
    return template


def build_page_instruct(page, num_page):
    template = f"This is the image on page {page + 1} of the {num_page} pages document. Describe the content in the page."
    return template


class DocIRDataLoader(Dataset):
    def __init__(self, root_path):
        self.data_dir = root_path
        self.dataset_df = pd.read_parquet(os.path.join(root_path, 'MMDocIR_doc_passages.parquet'))
        self.data_json = []
        for line in open(os.path.join(root_path, "MMDocIR_gt_remove.jsonl"), 'r', encoding="utf-8"):
            self.data_json.append(json.loads(line.strip()))
        if os.path.exists(save_csv_file):
            self.infered_data = pd.read_csv(save_csv_file)
            self.infered_questions = list(self.infered_data['question_id'])
        else:
            self.infered_questions = []

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        if idx + 10000 in self.infered_questions:
            return None
        file_info = self.data_json[idx]
        doc_name = file_info["doc_name"]
        doc_pages = self.dataset_df.loc[self.dataset_df['doc_name'] == doc_name]
        res = {"question_id": file_info['question_id'], "queries": file_info['question'], "passages": [],
               'doc_name': doc_name}
        return res

    def get_by_question_id(self, question_id):
        file_info = self.data_json[question_id]
        doc_name = file_info["doc_name"]
        doc_pages = self.dataset_df.loc[self.dataset_df['doc_name'] == doc_name]
        res = {"question_id": file_info['question_id'], "queries": file_info['question'], "passages": [],
               }
        for page in doc_pages.itertuples():
            tmp = {"page_id": page.passage_id, 'image_content': Image.open(io.BytesIO(page.image_binary))}
            res['passages'].append(tmp)
        return res

    def get_loader(self):
        return DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=5,
                          collate_fn=lambda x: x[0])


class Retriever:
    def __init__(self, model_path, loader, datasets):
        self.dataset = datasets
        self.gme = GmeQwen2VL(
            model_name="gme-Qwen2-VL-7B-Instruct",
            model_path=model_path,
            device="cuda:0"
        )
        self.loader = loader

    def eval(self, ):
        model = self.gme
        ans_with_score = {}
        if os.path.exists(save_csv_file):
            res = pd.read_csv(save_csv_file).to_dict('records')
        else:
            res = []
        old_doc_name = "None"
        for json_data in tqdm(loader):
            try:
                if json_data is None:
                    continue
                cur_doc_name = json_data['doc_name']
                images = [item['image_content'] for item in json_data['passages']]
                images_ids = [item['page_id'] for item in json_data['passages']]
                queries = [json_data['queries']]
                batch_queries = model.get_text_embeddings(
                    texts=queries,
                    instruction=get_instruction(queries[0])
                ).cpu()
                tot_pages = len(json_data['passages'])
                instructions = [build_page_instruct(int(item['page_id']), tot_pages) for item in json_data['passages']]
                if cur_doc_name != old_doc_name:
                    batch_doc_embeddings, batch_doc_idx2page = self.get_cached_embeddings(doc_name=json_data['doc_name'])
                    if batch_doc_idx2page is not None:
                        batch_doc_embeddings = torch.cat(batch_doc_embeddings).cpu()
                if batch_doc_embeddings is not None:
                    scores = (batch_queries * batch_doc_embeddings).sum(-1)
                    answers = torch.argsort(scores, descending=True)[:30]
                    sorted_values = scores[answers]
                    answers = [batch_doc_idx2page[idx] for idx in answers]
                    str_ans = [f'"{int(x)}"' for x in answers]
                    formatted_str = "[" + ", ".join(str_ans) + "]"
                    tmp = {"question_id": json_data['question_id'], "passage_id": formatted_str}

                ans_with_score[tmp['question_id']] = {
                    "passage_id": tmp['passage_id'],
                    "scores": sorted_values.tolist()
                }
                res.append(tmp)
                df = pd.DataFrame(res)
                df.to_csv(save_csv_file, index=False)
                old_doc_name = cur_doc_name
            except Exception as e:
                print(f"处理失败: {str(e)}, question_id: {json_data['question_id']}")
                zeros = ["0"] * 30
                str_ans = [f'"{x}"' for x in zeros]
                formatted_str = "[" + ", ".join(str_ans) + "]"
                tmp = {"question_id": json_data['question_id'], "passage_id": formatted_str}
                ans_with_score[tmp['question_id']] = {
                    "passage_id": tmp['passage_id'],
                    "scores": [0] * 30
                }
                res.append(tmp)
                df = pd.DataFrame(res)
                df.to_csv(save_csv_file, index=False)
                
        with open(save_json_file, 'w', newline='') as j_file:
            json.dump(ans_with_score, j_file)

    def get_cached_embeddings(self, doc_name):
        if not os.path.exists(f'./embed_store_33/{doc_name}.pkl'):
            return None, None
        with open(f'./embed_store_33/{doc_name}.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['batch_embeddings'], data['page_ids']


if __name__ == '__main__':
    loader = DocIRDataLoader(os.environ.get("MMDOCIR_ROOT")).get_loader()
    re = Retriever(model_path=os.environ.get("GME_PATH"), loader=loader,
                   datasets=None)
    re.eval()
