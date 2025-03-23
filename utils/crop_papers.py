# 由于执行状态已重置，需重新加载 OpenCV 并读取图片
import cv2
import numpy as np
from PIL import Image


# 重新定义 layout 提取函数
def extract_layout_info(img):
    """
    使用 OpenCV 提取论文截图的版面布局信息（段落、标题、表格、图像）
    1. 进行边缘检测
    2. 轮廓检测并过滤文本块
    3. 绘制 bounding boxes 进行可视化
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行二值化（自适应阈值）
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    # 进行膨胀操作，使文本块更加连贯
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # 轮廓检测
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤小块区域，并绘制 bounding boxes
    layout_img = img.copy()
    detected_regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:  # 过滤小噪声区域
            cv2.rectangle(layout_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected_regions.append((x, y, w, h))

    return layout_img, detected_regions


# 重新定义裁剪函数
def extract_detected_regions(img, regions):
    """
    从检测到的布局区域中裁剪出对应部分
    :param img: 原始图像
    :param regions: 识别出的(x, y, w, h)区域列表
    :return: 裁剪后的区域字典（文件路径 -> 图像）
    """
    cropped_images = {}
    pil_cropped_images = []
    for i, (x, y, w, h) in enumerate(regions):
        cropped_img = img[y:y + h, x:x + w]  # 进行裁剪
        # cropped_path = f"./paper_crop/cropped_region_{i}.png"
        # cv2.imwrite(cropped_path, cropped_img)  # 保存裁剪图片
        # cropped_images[cropped_path] = cropped_img
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))  # 转换为 PIL Image 格式
        pil_cropped_images.append(pil_img)  # 存入字典
    return pil_cropped_images


def get_paper_layout(img_table):
    # # 重新加载图片路径
    # image_path = './store_page/img_29.png'
    # 读取论文截图
    # img_table = cv2.imread(image_path)
    # 进行 layout 信息提取
    layout_img_table, detected_regions_table = extract_layout_info(img_table)
    # 裁剪检测到的区域
    cropped_regions_table = extract_detected_regions(img_table, detected_regions_table)
    return cropped_regions_table
