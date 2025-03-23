import os
import json
import pandas as pd
from functools import lru_cache
from torch.utils.data import Dataset


class MMDocIRDataset(Dataset):
    def __init__(
        self, 
        question_path=os.environ["MMDOCIR_QUESTION_PATH"], 
        passage_path=os.environ["MMDOCIR_PASSAGE_PATH"], 
        passage_img_dir=os.environ["MMDOCIR_ROOT"]
    ):
        self.question_path = question_path
        self.passage_img_dir = passage_img_dir
        self.passage_df = pd.read_parquet(passage_path)
        self.data_items = []
        for line in open(self.question_path, 'r', encoding="utf-8"):
            self.data_items.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data_items)
    
    @lru_cache(maxsize=10000)
    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self.m2kr_passages)
            step = index.step if index.step is not None else 1
            return [self[i] for i in range(start, stop, step)]
        data_item = self.data_items[index]
        data = {
            "question_id": data_item["question_id"],
            "question": data_item["question"],
            "doc_name": data_item["doc_name"],
            "targets": [
                {
                    "passage_id": row["passage_id"],
                    "image_path": os.path.join(self.passage_img_dir, row["image_path"]),
                    "ocr_text": row["ocr_text"],
                    # "vlm_text": row["vlm_text"]
                }
                for _, row in self.passage_df.loc[self.passage_df['doc_name'] == data_item["doc_name"]].iterrows()
            ]
        }
        return data
    
    def __getslice__(self, start, end):
        return [self[i] for i in range(start, end)]
    

if __name__ == "__main__":
    dataset = MMDocIRDataset()
    print(f"MMDocIRDataset Count: {len(dataset)}")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))