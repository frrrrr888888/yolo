import itertools
import random
import cv2
import os
import glob
import numpy as np
import yaml
import shutil
from transform import *
import re
import math

class DotDict(dict):
    """A dictionary that supports dot notation as well as dictionary access notation."""

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_dict(data):
        """Recursively converts nested dictionaries to DotDicts."""
        if isinstance(data, dict):
            return DotDict({k: DotDict.from_dict(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [DotDict.from_dict(i) for i in data]
        else:
            return data

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

def yaml2dotdict(yaml_file, key=False):
    config = yaml_load(yaml_file)
    if key:
        dot_data = DotDict.from_dict(config[key])
    else:
        dot_data = DotDict.from_dict(config)
    return dot_data

def clean_bbox(x_c, y_c, w, h):
    """
    清洗坐标：解决浮点数计算导致的越界问题
    """
    x_min = np.float64(x_c - w / 2)
    y_min = np.float64(y_c - h / 2)
    x_max = np.float64(x_c + w / 2)
    y_max = np.float64(y_c + h / 2)

    x_min = np.float64(max(0.0, min(1.0, x_min)))
    y_min = np.float64(max(0.0, min(1.0, y_min)))
    x_max = np.float64(max(0.0, min(1.0, x_max)))
    y_max = np.float64(max(0.0, min(1.0, y_max)))

    w_new = np.float64(x_max - x_min)
    h_new = np.float64(y_max - y_min)
    x_c_new = np.float64(x_min + w_new / 2)
    y_c_new = np.float64(y_min + h_new / 2)

    return x_c_new, y_c_new, w_new, h_new

def calculate_corners(x_c, y_c, w, h):
    """
    计算框的四个角坐标
    """
    # 计算四个角的坐标
    top_left = (x_c - w / 2, y_c - h / 2)
    top_right = (x_c + w / 2, y_c - h / 2)
    bottom_left = (x_c - w / 2, y_c + h / 2)
    bottom_right = (x_c + w / 2, y_c + h / 2)
    return np.array([top_left, top_right, bottom_left, bottom_right])

def clean_by_index(index, keypoints):
    # 使用 index 中的下标来过滤 keypoints，先将 index 中的 float 转换为 int
    cleaned_keypoints = [keypoints[int(i)] for i in index if int(i) < len(keypoints)]
    return cleaned_keypoints

def flip_image(image, bboxes, keypoints, should_flip=True, flip_prob=0.5):
    """
    随机进行水平或垂直翻转，并更新目标框和关键点坐标。

    :param image: 输入的图像
    :param bboxes: 包含图像目标框的列表，每个框是 [x_c, y_c, w, h]，归一化到[0, 1]
    :param keypoints: 关键点的列表，每个关键点是 (x, y, class_id)，归一化到[0, 1]
    :param should_flip: 是否执行翻转（默认：True）
    :param flip_prob: 执行翻转的概率（默认：50%）

    :return: 翻转后的图像、目标框和关键点
    """
    if should_flip and random.random() < flip_prob:  # 按照概率选择是否翻转
        # 随机选择翻转类型：水平翻转或垂直翻转
        flip_type = random.choice(["horizontal", "vertical"])

        if flip_type == "horizontal":
            # 水平翻转
            flipped_image = cv2.flip(image, 1)
            flipped_bboxes = []
            flipped_keypoints = []

            # 更新目标框和关键点
            for bbox, kp_list in zip(bboxes, keypoints):
                # 翻转目标框：x_c -> 1 - x_c
                x_c, y_c, w, h = bbox
                flipped_bbox = [1 - x_c, y_c, w, h]  # 水平翻转时只改变 x_c

                # 翻转关键点
                flipped_kps = []
                for x, y, k_cls in kp_list:
                    if x == 0 and y == 0:
                        flipped_kps.append([x, y, k_cls])
                    else:
                        flipped_kps.append([1 - x, y, k_cls])  # 水平翻转时只改变 x

                flipped_bboxes.append(flipped_bbox)
                flipped_keypoints.append(flipped_kps)

        elif flip_type == "vertical":
            # 垂直翻转
            flipped_image = cv2.flip(image, 0)
            flipped_bboxes = []
            flipped_keypoints = []

            # 更新目标框和关键点
            for bbox, kp_list in zip(bboxes, keypoints):
                # 翻转目标框：y_c -> 1 - y_c
                x_c, y_c, w, h = bbox
                flipped_bbox = [x_c, 1 - y_c, w, h]  # 垂直翻转时只改变 y_c

                # 翻转关键点
                flipped_kps = []
                for x, y, k_cls in kp_list:
                    if x == 0 and y == 0:
                        flipped_kps.append([x, y, k_cls])
                    else:
                        flipped_kps.append([x, 1 - y, k_cls])  # 垂直翻转时只改变 y

                flipped_bboxes.append(flipped_bbox)
                flipped_keypoints.append(flipped_kps)

    else:
        # 如果不进行翻转，则返回原图和原始数据
        flipped_image = image
        flipped_bboxes = bboxes
        flipped_keypoints = keypoints

    return flipped_image, flipped_bboxes, flipped_keypoints

def shape_size(list0, size):
    x, y, w, h = list0

    x *= size[1]
    y *= size[0]
    w *= size[1]
    h *= size[0]

    return [x, y, w, h]

def check_in(x, y, width, height):
    """
    判断点 (x, y) 是否在图像范围内

    参数:
        x, y: 坐标（可以是 float）
        width, height: 图像尺寸

    返回:
        True / False
    """
    return 0 <= x < width and 0 <= y < height

def is_gray(image, x, y, tol=5):
    """
    判断某个点是否为接近(114,114,114)的灰色

    参数：
        image: cv2图像 (BGR)
        x, y: 像素坐标
        tol: 容差（默认±5）

    返回：
        True / False
    """
    h, w = image.shape[:2]

    x *= w
    y *= h

    # 防止越界
    if not (0 <= x < w and 0 <= y < h):
        return False

    b, g, r = image[int(y), int(x)]

    return (
        abs(int(b) - 114) <= tol and
        abs(int(g) - 114) <= tol and
        abs(int(r) - 114) <= tol
    )

def image_draw(
    image,
    bboxes,
    keypoints,
    box_color=(0, 255, 0),
    point_color=(0, 0, 255),
    thickness=2,
    radius=5
):
    """
    在图像上绘制 bbox 和 keypoints

    参数：
        image: 输入图像 (H, W, C)
        bboxes: YOLO格式 [(x_c, y_c, w, h), ...] 归一化坐标
        keypoints: [[(x, y, v), ...], ...] 归一化坐标
        box_color: bbox颜色 (B, G, R)
        point_color: 关键点颜色 (B, G, R)
        thickness: bbox线宽
        radius: 关键点半径

    返回：
        绘制后的图像
    """

    h, w = image.shape[:2]

    for bbox, kp in zip(bboxes, keypoints):
        # === bbox ===
        x_c, y_c, bw, bh = bbox

        x_min = int((x_c - bw / 2) * w)
        y_min = int((y_c - bh / 2) * h)
        x_max = int((x_c + bw / 2) * w)
        y_max = int((y_c + bh / 2) * h)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, thickness)

        # === keypoints ===
        for (key_x, key_y, _) in kp:
            px = int(key_x * w)
            py = int(key_y * h)

            cv2.circle(image, (px, py), radius, point_color, -1)

    return image

def get_dataset_paths(root_path, out_path):

    """示例：
    文件夹格式：8位日期20220101\images\train\imgs"""

    INPUT_IMG_DIRS = []
    INPUT_LABEL_DIRS = []
    OUTPUT_IMG_DIRS = []
    OUTPUT_LABEL_DIRS = []
    number = []

    # 匹配：8位数字开头（日期）
    pattern = re.compile(r"^\d{8}.*")

    for name in os.listdir(root_path):
        full_path = os.path.join(root_path, name)
        dst_path = os.path.join(out_path, name)

        # 只要符合“日期开头 + 是文件夹”
        if os.path.isdir(full_path) and pattern.match(name):
            img_dir = os.path.join(full_path, "images", "train")
            label_dir = os.path.join(full_path, "labels", "train")
            out_img_dir = os.path.join(dst_path + '_aug', "images", "train")
            out_label_dir = os.path.join(dst_path + '_aug', "labels", "train")

            # 可选：判断路径是否存在
            if os.path.exists(img_dir) and os.path.exists(label_dir):

                os.makedirs(out_img_dir, exist_ok=True)
                os.makedirs(out_label_dir, exist_ok=True)

                INPUT_IMG_DIRS.append(img_dir)
                INPUT_LABEL_DIRS.append(label_dir)
                OUTPUT_IMG_DIRS.append(out_img_dir)
                OUTPUT_LABEL_DIRS.append(out_label_dir)
            else:
                print(f"[WARNING] 缺少路径: {full_path}")
            # 统计图片
            imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            if not len(imgs) == len(labels):
                print(f"[WARNING] 地址，图片数量，label数量: {full_path, len(imgs), len(labels)}")
            number.append(len(imgs))

    return INPUT_IMG_DIRS, INPUT_LABEL_DIRS, OUTPUT_IMG_DIRS, OUTPUT_LABEL_DIRS, number

def compute_scales(nums, max_scale=10000):
    #计算每个数据集的方法倍数
    if not nums:
        return []

    max_val = max(nums)*2
    scales = []

    if len(nums) == 1:
        scales.append(2)
        return scales

    for x in nums:
        if x == 0:
            scales.append(0)
        else:
            scale = max_val / x
            scale = max(1, min(scale, max_scale))  # 限制范围
            scales.append(scale)

    return scales

def match_scales(paths, date_scale_list=None, aug_scale=1):

    result = []

    if date_scale_list is None:
        for _ in paths:
            result.append(aug_scale)
    else:
        date_to_scale = {str(d): s for d, s in date_scale_list}

        for p in paths:
            parts = p.replace("\\", "/").split("/")  # 兼容Windows路径

            scale = 0

            if len(parts) >= 3:
                folder_name = parts[-3]   #  20260101_xxx/images/tain
                date_prefix = folder_name[:8]  #  20260101

                scale = date_to_scale.get(date_prefix, aug_scale)
                if scale < 1:
                    raise ValueError(f"❌ scale < 1: {scale}, path: {p}")

            result.append(scale)

    return result

def process_image(
    image,
    bboxes,
    class_labels,
    keypoints,
    indexes,
    transform,
    OUTPUT_IMG_DIR,
    OUTPUT_LABEL_DIR,
    base_name,
    i,
    should_flip=False,
    flip_prob=0.5,
    should_draw=False
):
    """
    image: 输入图像 (H, W, C)
    bboxes: 目标框列表，每个为 [x_c, y_c, w, h]（归一化）
    class_labels: 每个目标框对应的类别标签
    keypoints: 每个目标的关键点，格式 [[(x, y, cls), ...], ...]
    indexes: 目标索引，用于增强后筛选有效目标
    transform: 数据增强函数（albumentations）
    OUTPUT_IMG_DIR: 增强后图像保存路径
    OUTPUT_LABEL_DIR: 增强后标签保存路径
    base_name: 原图名称（不含后缀）
    i: 当前增强编号
    should_flip: 是否进行翻转
    flip_prob: 翻转概率
    should_draw: 是否绘制可视化结果
    """

    # ===== 1. 保存原始bbox（避免被transform修改）=====
    orig_bboxes = bboxes.copy()

    # ===== 2. 执行增强 =====
    transformed = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels,
        indexes=indexes
    )

    aug_img = transformed['image']
    aug_bboxes = transformed['bboxes']
    aug_labels = transformed['class_labels']
    aug_indexes = transformed['indexes']

    if len(aug_bboxes) == 0:
        return

    # ===== 3. 用“最后一个框”计算全局单应矩阵 H =====
    ref_aug = aug_bboxes[-1]
    ref_orig = orig_bboxes[-1]

    x_c, y_c, w, h = shape_size(ref_aug, image.shape)
    ox, oy, ow, oh = shape_size(ref_orig, image.shape)

    H = cv2.getPerspectiveTransform(
        np.float32(calculate_corners(ox, oy, ow, oh)),
        np.float32(calculate_corners(x_c, y_c, w, h))
    )

    # ===== 4. 去掉参考框 =====
    aug_bboxes = aug_bboxes[:-1]
    aug_labels = aug_labels[:-1]
    aug_indexes = aug_indexes[:-1]

    if len(aug_bboxes) == 0:
        return

    # ===== 5. 根据索引同步过滤 =====
    aug_keypoints = clean_by_index(aug_indexes, keypoints)
    orig_bboxes = clean_by_index(aug_indexes, orig_bboxes)

    # ===== 6. 关键点变换 =====
    augmented_keypoints = []

    for kp_group in aug_keypoints:
        updated_kps = []

        for (x, y, k_cls) in kp_group:

            # 6.1 反归一化
            x *= image.shape[1]
            y *= image.shape[0]

            if x == 0 and y == 0:
                new_x, new_y = 0.0, 0.0
            else:
                # 6.2 单应变换
                kp_vec = np.array([x, y, 1.0])
                new_kp = H @ kp_vec

                new_x = new_kp[0] / new_kp[2]
                new_y = new_kp[1] / new_kp[2]

            # 6.3 边界检查
            if check_in(new_x, new_y, image.shape[1], image.shape[0]):
                new_x /= image.shape[1]
                new_y /= image.shape[0]
            else:
                new_x, new_y = 0.0, 0.0

            # 6.4 灰度过滤
            if is_gray(aug_img, new_x, new_y):
                updated_kps.append((0, 0, k_cls))
            else:
                updated_kps.append((new_x, new_y, k_cls))

        augmented_keypoints.append(updated_kps)

    # ===== 7. 可选翻转 =====
    aug_img, aug_bboxes, augmented_keypoints = flip_image(
        aug_img, aug_bboxes, augmented_keypoints, should_flip, flip_prob
    )

    # ===== 8. 可视化 =====
    if should_draw:
        aug_img = image_draw(aug_img, aug_bboxes, augmented_keypoints)

    # ===== 9. 保存 =====
    save_name = f"{base_name}_aug_{i}"

    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, save_name + ".png"), aug_img)

    with open(os.path.join(OUTPUT_LABEL_DIR, save_name + ".txt"), "w") as f:
        for cls, bbox, kp in zip(aug_labels, aug_bboxes, augmented_keypoints):

            x_c, y_c, w, h = clean_bbox(*bbox)

            line = f"{int(cls)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

            for (kx, ky, kc) in kp:
                line += f" {kx:.6f} {ky:.6f} {int(kc)}"

            f.write(line + "\n")

def generate_scale_list(scales, n):
    """
    scales: 放大倍数（int 或 float）
    n: len(img_paths)

    return: 长度为 n 的 list
    """

    if n == 0:
        return []

    base = math.floor(scales)  # 向下取整
    result = [base] * n

    # 如果是整数，直接返回
    if scales == base:
        return result

    # 计算需要 +1 的个数
    extra = int(math.ceil(scales * n - base * n))

    # 给前 extra 个位置 +1
    for i in range(extra):
        result[i] += 1

    return result

def augment_data(path):
    config = yaml2dotdict(path)

    # 路径配置
    data_path = config.data_path
    out_path = config.out_path
    INPUT_IMG_DIRS, INPUT_LABEL_DIRS, OUTPUT_IMG_DIRS, OUTPUT_LABEL_DIRS, number = get_dataset_paths(data_path, out_path)
    auto_path = config.auto_path
    auto = config.auto
    if auto_path:
        scales = compute_scales(number)
    else:
        aug_scale = config.aug_scale
        if config.Unified_settings:
            scales = match_scales(INPUT_IMG_DIRS,aug_scale=aug_scale)
        else:
            path_scale = config.path_scale
            scales = match_scales(INPUT_IMG_DIRS, path_scale)
            #scales = match_scales(INPUT_IMG_DIRS, path_scale, aug_scale)

    should_flip = config.should_flip
    flip_prob = config.flip_prob
    should_draw = config.should_draw
    category_map = config.category_map

    for num, (INPUT_IMG_DIR, INPUT_LABEL_DIR, OUTPUT_IMG_DIR, OUTPUT_LABEL_DIR) in enumerate(zip(INPUT_IMG_DIRS, INPUT_LABEL_DIRS,
                                                                                                 OUTPUT_IMG_DIRS,OUTPUT_LABEL_DIRS)):
        scale = scales[num] - 1

        img_paths = glob.glob(os.path.join(INPUT_IMG_DIR, '*.png'))
        if auto:
            print(f"{INPUT_IMG_DIRS[num]}找到 {len(img_paths)} 张图片，开始自动增强...")
        else:
            print(f"{INPUT_IMG_DIRS[num]}找到 {len(img_paths)} 张图片，开始增强{scale + 1}倍...")

        scale_list = generate_scale_list(scale, len(img_paths))
        number_images = 0

        for img_path in img_paths:
            image = cv2.imread(img_path)
            if image is None: continue

            base_name = os.path.basename(img_path).replace('.png', '')
            txt_path = os.path.join(INPUT_LABEL_DIR, base_name + '.txt')
            dst_text_path = os.path.join(OUTPUT_LABEL_DIR, base_name + '.txt')

            shutil.copy(txt_path, dst_text_path)

            if not os.path.exists(txt_path): continue

            bboxes = []  # 用于存储原始的所有目标框数据
            class_labels = []
            indexes = []  # 存储每个框的索引
            keypoints = []  # 用于存储每个框的关键点

            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    data = line.strip().split()
                    if not data: continue

                    cls = int(data[0])
                    raw_bbox = [float(x) for x in data[1:5]]  # [x_c, y_c, w, h]
                    raw_keypoints = [float(x) for x in data[5:]]  # 关键点是后面的数据

                    x_c, y_c, w, h = clean_bbox(raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3])

                    if w < 0.0001 or h < 0.0001: continue

                    # 提取关键点
                    kp = []
                    for i in range(0, len(raw_keypoints), 3):  # 每三个单位是一个(x, y, k_cls)
                        kp.append((raw_keypoints[i], raw_keypoints[i + 1], raw_keypoints[i + 2]))

                    bboxes.append([x_c, y_c, w, h])
                    class_labels.append(cls)
                    indexes.append(idx)  # 记录框的索引
                    keypoints.append(kp)  # 记录框的关键点

                raw_img = image.copy()
                if should_draw:
                    raw_img = image_draw(
                        raw_img,
                        bboxes,
                        keypoints,
                        box_color=(0, 255, 0),
                        point_color=(255, 0, 0),
                    )

                # 多存入一个正中心的框作为标致
                bboxes.append([0.5, 0.5, 0.1, 0.1])
                class_labels.append(0)
                indexes.append(len(indexes))  # 记录框的索引
                keypoints.append([0, 0, 0])  # 记录框的关键点

                cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, base_name + '.png'), raw_img)

            # 构造变换操作
            transform, AUGMENT_COUNT = build_transforms(augmentation_operations = config.augmentation_operations,
                                                        bbox_params_config = config.bbox_params,
                                                        a_count= scale_list[number_images],
                                                        bboexes = bboxes[:-1],
                                                        labels = class_labels[:-1],
                                                        category_map = category_map,
                                                        auto = auto)

            number_images += 1

            # 生成增强图
            for i in range(AUGMENT_COUNT):
                process_image(image,bboxes,class_labels,keypoints,indexes,transform,
                              OUTPUT_IMG_DIR,OUTPUT_LABEL_DIR,base_name,i,
                              should_flip=should_flip,flip_prob=flip_prob,should_draw=should_draw)

    print("增强完成！")


if __name__ == "__main__":
    augment_data(r"/ObjectRecognition/utils/datasets_preprocess/image_agument/config/rf.yaml")