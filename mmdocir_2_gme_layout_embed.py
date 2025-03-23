import io
import os
import pickle
from itertools import islice

import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import chromadb
from chromadb.api.types import IncludeEnum
from tqdm import tqdm
import ray
import utils.crop_papers
from models.gme import GmeQwen2VL

# 配置相关路径
docir_root = os.environ.get("MMDOCIR_ROOT")
model_path = os.environ.get("GME_PATH")
dataset_df = pd.read_parquet(os.environ.get("MMDOCIR_PASSAGE_PATH"))
doc_names = list(set(dataset_df['doc_name']))[:]
if not os.path.exists('./embed_store_33'):
    os.makedirs('./embed_store_33')
doc_names = [doc_name for doc_name in doc_names if not os.path.exists(f'./embed_store_33/{doc_name}.pkl')]


def build_page_instruct(page, num_page):
    template = f"This is the image on page {page + 1} of the {num_page} pages document. Describe the content in the page."
    return template


@ray.remote(num_gpus=1)
class PassageWorker:
    def __init__(self, model_path):
        # 加载模型（指定使用 GPU）
        self.gme_model = GmeQwen2VL(model_path, device='cuda')
        self.dataset_df = pd.read_parquet(os.environ.get("MMDOCIR_PASSAGE_PATH"))

    def clip_image(self, image):
        """将图像按照高度 800 进行切片"""
        clip_height = 800
        width, height = image.size
        num_clips = (height + clip_height - 1) // clip_height  # 向上取整
        clip_images = []
        for i in range(num_clips):
            top = i * clip_height
            bottom = min((i + 1) * clip_height, height)
            cropped_img = image.crop((0, top, width, bottom))
            clip_images.append(cropped_img)
        return clip_images

    def process_batch(self, img_batch):
        """使用模型计算一批图像的嵌入"""
        with torch.no_grad():
            embeddings = self.gme_model.get_image_embeddings(img_batch).cpu()
        return [embeddings[i, :].tolist() for i in range(len(img_batch))]

    def get_batch_embeddings(self, images, instructions):
        st = 0
        tot_images = []
        idx2page_id = []
        texts = []
        # 首先是整个页面图的embedding
        for i in range(len(images)):
            page_images = [images[i]]
            page_images.extend(utils.crop_papers.get_paper_layout(np.array(images[i])))
            tot_images.extend(page_images)
            idx2page_id.extend([i] * len(page_images))
            texts.extend([instructions[i]] * len(page_images))
        batch_size = 1
        tot_embeddings = []
        while True:
            with torch.no_grad():
                batch_images = tot_images[st:st + batch_size]
                batch_instruct = texts[st:st + batch_size]
                batch_embedding = self.gme_model.get_fused_embeddings(texts=batch_instruct, images=batch_images).cpu()
                tot_embeddings.append(batch_embedding)
            st += batch_size
            if st >= len(tot_images): break
        return tot_embeddings, idx2page_id

    def process_passage(self, doc_name, batch_size=1):
        """
        处理单个 Passage：
         1. 加载图像，并切片
         2. 按照 batch_size 分批计算图像嵌入
         3. 每处理完一个 batch，就实时调用 writer 写入
         4. 返回：处理信息以及所有写入任务的 ObjectRef 列表
        """
        writer_tasks = []  # 用于保存写入任务的 ObjectRef
        doc_pages = self.dataset_df.loc[self.dataset_df['doc_name'] == doc_name]
        pages = []
        for page in doc_pages.itertuples():
            tmp = {"page_id": page.passage_id, 'image_content': Image.open(io.BytesIO(page.image_binary))}
            pages.append(tmp)
        tot_pages = len(pages)
        images = [item['image_content'] for item in pages]
        instructions = [build_page_instruct(int(item['page_id']), tot_pages) for item in pages]
        batch_doc_embeddings, batch_doc_idx2page = self.get_batch_embeddings(images, instructions)
        res = {"page_ids": batch_doc_idx2page, "batch_embeddings": batch_doc_embeddings}

        with open(f'./embed_store_33/{doc_name}.pkl', 'wb') as f:
            pickle.dump(res, f)
        print(f"Good! Passage {doc_name} processed.", writer_tasks)
        return f"Passage {doc_name} processed.", writer_tasks


def main(num_workers=8):
    ray.init()

    # 创建 PassageWorker actor 池，每个 actor 占用 1 个 GPU
    workers = [
        PassageWorker.remote(model_path)
        for _ in range(num_workers)
    ]

    passage_tasks = []
    # 遍历所有 Passage，采用轮询方式分配任务给各个 worker
    for i, doc_name in tqdm(enumerate(doc_names), total=len(doc_names), desc="Submitting tasks"):
        # passage_id = row['passage_id']
        # page_path = row['page_screenshot']
        passage_tasks.append(workers[i % num_workers].process_passage.remote(doc_name))

    writer_task_refs = []  # 收集所有 writer 任务的 ObjectRef
    with tqdm(total=len(passage_tasks), leave=True, desc="Processing passages") as pbar:
        for task in ray.get(passage_tasks):
            msg, writer_refs = task
            # print(msg)
            writer_task_refs.extend(writer_refs)
            pbar.update(1)

    # 等待所有 writer 写入任务完成
    with tqdm(total=len(writer_task_refs), desc="Writing embeddings") as pbar:
        remaining = writer_task_refs.copy()
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            for ref in done:
                print(ray.get(ref))
                pbar.update(1)

    ray.shutdown()


if __name__ == '__main__':
    # print(dataset_df)
    main()
