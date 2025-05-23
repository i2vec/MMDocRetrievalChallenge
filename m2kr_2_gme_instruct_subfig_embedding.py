import os
from itertools import islice

import pandas as pd
import PIL.Image as Image
import torch
import chromadb
from tqdm import tqdm

from models.gme import GmeQwen2VL
from utils.extract_image import extract_sub_image

# 配置相关路径
m2kr_root = os.environ.get("M2KR_ROOT")
passage_path = os.environ.get("M2KR_PASSAGES_PATH")
m2kr_challenge_img_root = os.environ.get("M2KR_PASSAGES_IMG_DIR")
chroma_db_path = "./chroma_db_2"
collection_name = "m2kr_image"
model_path = os.environ.get("GME_PATH")


class PassageWorker:
    def __init__(self, model_path, challenge_img_root):
        # 加载模型（指定使用 GPU）
        self.gme_model = GmeQwen2VL(model_path, device='cuda')
        self.challenge_img_root = challenge_img_root

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

    def batch_iterator(self, iterable, batch_size):
        """将 iterable 按照 batch_size 分批"""
        it = iter(iterable)
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            yield batch

    def process_batch(self, img_batch):
        """使用模型计算一批图像的嵌入"""
        with torch.no_grad():
            embeddings = self.gme_model.get_image_embeddings(img_batch).cpu()
        return [embeddings[i, :].tolist() for i in range(len(img_batch))]

    def process_passage(self, passage_id, page_path, writer, batch_size=1):
        """
        处理单个 Passage：
         1. 加载图像，并切片
         2. 按照 batch_size 分批计算图像嵌入
         3. 每处理完一个 batch，就实时调用 writer 写入
         4. 返回：处理信息
        """
        image_path = os.path.join(self.challenge_img_root, page_path)

        try:
            img = Image.open(image_path)
        except Exception as e:
            err_msg = f"Passage {passage_id} error in opening image: {e}"
            print(err_msg)
            return err_msg

        # 对图像进行切片
        batch_images = self.clip_image(img)[0:1]
        sub_images = extract_sub_image(image_path)
        batch_images.extend(sub_images)
        current_idxs = list(range(len(batch_images)))
        all_embeddings = []
        # 分批处理图像
        for img_batch in self.batch_iterator(batch_images, batch_size):
            all_embeddings.extend(self.process_batch(img_batch))
            
        # 写入嵌入
        writer.add_embeddings(passage_id, current_idxs, all_embeddings)
        return f"Passage {passage_id} processed."


class ChromaWriter:
    def __init__(self, chroma_db_path, collection_name, tot_num):
        # 仅由单一 writer 负责写入，避免并发写入问题
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.tot_num = tot_num
        self.cnt = 0

    def add_embeddings(self, passage_id, idxs, embeddings):
        """
        写入嵌入：
         根据 passage_id 和下标生成唯一 ID，然后写入嵌入及文档内容（这里将 ID 作为文档）
        """
        docs = [f"{passage_id}_{i}" for i in idxs]
        self.collection.add(
            ids=docs,
            documents=docs,
            embeddings=embeddings
        )
        self.cnt += 1
        print(f"finished adding {passage_id}")
        print(f"finished  {self.cnt} / {self.tot_num} which is {self.cnt / self.tot_num}%")
        return f"Writer: Added embeddings for {passage_id} indices {idxs}"


def main():
    # 读取 Passage 数据
    df_passages = pd.read_parquet(passage_path)
    if os.environ["DEBUG"] == "true":
        df_passages = df_passages[:20]

    # 创建 PassageWorker 实例
    worker = PassageWorker(model_path, m2kr_challenge_img_root)

    # 创建 writer 实例
    writer = ChromaWriter(chroma_db_path, collection_name, len(df_passages))

    # 遍历所有 Passage 处理
    for i, row in tqdm(df_passages.iterrows(), total=len(df_passages), desc="Processing passages"):
        passage_id = row['passage_id']
        page_path = row['page_screenshot']
        msg = worker.process_passage(passage_id, page_path, writer)
        print(msg)


if __name__ == '__main__':
    main()
