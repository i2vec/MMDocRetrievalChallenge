import cv2
import os

from PIL import Image


def detect_image_regions(image_path, min_size=50):
    """
    自动检测图像中可能的图片区域
    使用 Canny 边缘检测 + 轮廓提取的方法，
    并过滤掉宽或高小于 min_size 的噪声区域
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片{image_path}，请检查路径")
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > min_size and h > min_size:
            boxes.append((x, y, w, h))
    return boxes


def get_area(box):
    """
    返回矩形框 box 的面积
    box 格式: (x, y, w, h)
    """
    _, _, w, h = box
    return w * h


def get_intersection_area(boxA, boxB):
    """
    计算两个框的交叠区域面积
    若无交叠则返回 0
    """
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB

    left = max(xA, xB)
    top = max(yA, yB)
    right = min(xA + wA, xB + wB)
    bottom = min(yA + hA, yB + hB)

    if right > left and bottom > top:
        return (right - left) * (bottom - top)
    else:
        return 0


def can_merge(boxA, boxB, overlap_threshold=0.5):
    """
    如果两个框重叠部分占较小框面积比例大于 overlap_threshold，则认为需要合并
    """
    intersection = get_intersection_area(boxA, boxB)
    if intersection == 0:
        return False
    smaller_area = min(get_area(boxA), get_area(boxB))
    return (intersection / smaller_area) > overlap_threshold


def merge_two_boxes(boxA, boxB):
    """
    合并两个框，返回外接矩形
    """
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    merged_left = min(xA, xB)
    merged_top = min(yA, yB)
    merged_right = max(xA + wA, xB + wB)
    merged_bottom = max(yA + hA, yB + hB)
    return (merged_left, merged_top, merged_right - merged_left, merged_bottom - merged_top)


def merge_overlapping_boxes(boxes, overlap_threshold=0.5):
    """
    对检测到的 boxes 进行多次迭代合并，
    当任意两个框重叠比例超过 overlap_threshold 时，
    用它们的外接矩形替换
    """
    merged = True
    while merged:
        merged = False
        new_boxes = []
        while boxes:
            box = boxes.pop(0)
            merged_box = box
            i = 0
            while i < len(new_boxes):
                nb = new_boxes[i]
                if can_merge(merged_box, nb, overlap_threshold):
                    merged_box = merge_two_boxes(merged_box, nb)
                    new_boxes.pop(i)
                    merged = True
                    # 合并后重头开始检测该框是否能继续与其他框合并
                    i = 0
                else:
                    i += 1
            new_boxes.append(merged_box)
        boxes = new_boxes
    return boxes


def filter_boxes_by_area(boxes, max_area=800000):
    """
    过滤掉面积超过 max_area 的区域
    """
    return [box for box in boxes if get_area(box) <= max_area]


def save_crops(image_path, boxes, output_folder="extracted_images"):
    """
    将每个 box 对应的区域从原图中裁剪并保存到指定文件夹
    """
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片，请检查路径")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    clip_images = []
    for idx, (x, y, w, h) in enumerate(boxes):
        crop = img[y:y + h, x:x + w]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        clip_images.append(pil_img)
    return clip_images
    # save_path = os.path.join(output_folder, f"crop_{idx}.png")
    # cv2.imwrite(save_path, crop)
    # print(f"已保存: {save_path}")


def extract_sub_image(image_path):
    # 1. 自动检测图片区域
    detected_boxes = detect_image_regions(image_path)
    # print("检测到的区域:")
    # for b in detected_boxes:
    #     print(b)
    # 3. 过滤掉像素面积超过 800000 的区域
    filtered_boxes = filter_boxes_by_area(detected_boxes, max_area=800000)
    # filtered_boxes = detected_boxes
    # print("过滤后的区域:")
    # for b in filtered_boxes:
    #     print(b)
    # 2. 对重叠过大的区域进行合并
    merged_boxes = merge_overlapping_boxes(filtered_boxes, overlap_threshold=0.6)
    # print("合并后的区域:")
    # for b in merged_boxes:
    #     print(b)
    # 4. 裁剪并保存结果
    clip_images = save_crops(image_path, merged_boxes)
    return clip_images


if __name__ == "__main__":
    image_path = "/mnt/vepfs/fs_ckps/xumj/data/wwwMrag/M2KR-Challenge/Challenge/Anthony_Pilkington.png"
    print(extract_sub_image(image_path))
