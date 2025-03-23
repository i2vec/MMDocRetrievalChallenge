import csv
import json
import os.path
import sys

import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.extract_image import extract_sub_image
from utils.image_similarity import cal_RANSAC_similarity

queries_path = os.environ.get("M2KR_QUERY_PATH")
m2kr_query_img_root = os.environ.get("M2KR_QUERY_IMG_DIR")
m2kr_root = os.environ.get("M2KR_ROOT")
passage_path = os.environ.get("M2KR_PASSAGES_PATH")
m2kr_challenge_img_root = os.environ.get("M2KR_PASSAGES_IMG_DIR")
chroma_db_path = "./chroma_db_2"
collection_name = "m2kr_image"
model_path = os.environ.get("GME_PATH")

df_passages = pd.read_parquet(passage_path)
df_queries = pd.read_parquet(queries_path)


def clip_image(image):
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


def save_image(idx, doc_name, img):
    if not os.path.exists(f'./tmp_retrieve/{doc_name}'):
        os.mkdir(f'./tmp_retrieve/{doc_name}')
    img.save(f'./tmp_retrieve/{doc_name}/{doc_name}_{idx}.png')


def store_screenshot(img_path, doc_name):
    tar_path = os.path.join(m2kr_challenge_img_root, img_path)
    try:
        image = Image.open(tar_path)
    except:
        print('failed to open {}'.format(tar_path))
        return
    batch_images = clip_image(image)[0:1]
    sub_images = extract_sub_image(tar_path)
    batch_images += sub_images
    for idx, img in enumerate(batch_images):
        save_image(idx, doc_name, img)


def use_similarity(question_id, passages):
    question_id = int(question_id)
    query_img = os.path.join(m2kr_query_img_root, df_queries.iloc[question_id].img_path)
    # query_image = Image.open(os.path.join(m2kr_query_img_root)).convert('RGB')
    res = []
    for item in passages:
        page_path = os.path.join('./tmp_retrieve', item.split('_')[0], f'{item}.png')
        try:
            tmp = cal_RANSAC_similarity(query_img, page_path)
        except:
            print('error occur')
            continue
        if tmp['Homography Inliers Ratio'] > 0.7:
            res.append(item.split('_')[0])
            # break
    unique_arr = list(dict.fromkeys(res))
    return unique_arr


def main():
    with open(os.environ.get("OUTPUT_FILE_M2KR_2"), 'r') as f:
        json_data = json.load(f)
    with open(os.environ.get("OUTPUT_FILE_M2KR_1"), 'r') as f:
        old_output = json.load(f)
    # old_output = pd.read_csv(os.environ.get("OUTPUT_FILE_M2KR_1"))
    cnt = 0
    with open(os.environ.get("OUTPUT_FILE_M2KR_3"), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['question_id', 'passage_id'])
        writer.writeheader()  # Write header row
        # m2kr
        for i, item in enumerate(tqdm(json_data)):
            passage_ids = item['fused_to_text']['passage_ids']
            scores = item['fused_to_text']['scores']
            # 比较像直接召回
            passage_ids = [page.split('_')[0] for idx, page in enumerate(passage_ids[:]) if
                           scores[idx] < 0.54]
            unique_arr = list(dict.fromkeys(passage_ids))
            if len(unique_arr) == 0:
                # 没有找到原图的case，但是子图包含
                unique_arr += use_similarity(item['question_id'], item['fused_to_text']['passage_ids'][:15])
                unique_arr = list(dict.fromkeys(unique_arr))
            if len(unique_arr) > 0: cnt += 1
            # 合并原有的答案
            passage_ids = unique_arr[:5] + old_output[i]['fused_to_text']['passage_id']
            passage_ids = list(dict.fromkeys(passage_ids))[:5]
            writer.writerow({
                'question_id': item['question_id'],
                'passage_id': json.dumps(passage_ids),
            })


if __name__ == '__main__':
    main()
    # merge_old_docir()
