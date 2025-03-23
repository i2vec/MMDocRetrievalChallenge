import os
import json
import pandas as pd
from torch.utils.data import Dataset

class M2KRQuestionDataset(Dataset):

    def __init__(
        self, 
        query_img_dir: str = os.environ["M2KR_QUERY_IMG_DIR"], 
        query_path: str = os.environ["M2KR_QUERY_PATH"]
    ):
        self.query_img_dir = query_img_dir
        self.query_path = query_path
        self.m2kr_queries = pd.read_parquet(query_path)

    def __len__(self):
        return len(self.m2kr_queries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self.m2kr_passages)
            step = index.step if index.step is not None else 1
            return [self[i] for i in range(start, stop, step)]
        
        data_item = self.m2kr_queries.iloc[index]
        return {
            "question_id": data_item['question_id'], 
            "question": data_item['question'], 
            "image_path": os.path.join(self.query_img_dir, data_item["img_path"]), 
        }




class M2KRPassagesDataset(Dataset):

    def __init__(
        self,
        m2kr_passages_path: str = os.environ["M2KR_PASSAGES_PATH"],
        m2kr_passages_img_dir: str = os.environ["M2KR_PASSAGES_IMG_DIR"]
    ):
        self.m2kr_passages_path = m2kr_passages_path
        self.m2kr_passages = pd.read_parquet(m2kr_passages_path)
        self.m2kr_passages_img_dir = m2kr_passages_img_dir
    def __len__(self):
        return len(self.m2kr_passages)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self.m2kr_passages)
            step = index.step if index.step is not None else 1
            return [self[i] for i in range(start, stop, step)]
        data_item = self.m2kr_passages.iloc[index]

        return {
            "passage_id": data_item['passage_id'], 
            "image_path": os.path.join(self.m2kr_passages_img_dir, data_item["page_screenshot"]), 
            "passage_content": data_item['passage_content']
        }

if __name__ == "__main__":
    dataset = M2KRPassagesDataset()
    print(f"M2KRPassagesDataset Count: {len(dataset)}")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    dataset = M2KRQuestionDataset()
    print(f"M2KRQuestionDataset Count: {len(dataset)}")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))