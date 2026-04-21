import random
import cv2
import os
import glob
import numpy as np
import yaml
import shutil
from transform import *


def read_yaml_config(file_path):
    """
    读取YAML配置文件

    Args:
        file_path: YAML文件路径

    Returns:
        dict: 配置字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        print("✅ YAML文件加载成功")
        return config

    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"❌ YAML解析错误: {e}")
        return {}
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return {}


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


def augment_data(path):
    config = read_yaml_config(path)

    # 路径配置
    INPUT_IMG_DIR = config['input_img_dir']
    INPUT_LABEL_DIR = config['input_label_dir']
    OUTPUT_IMG_DIR = config['output_img_dir']
    OUTPUT_LABEL_DIR = config['output_label_dir']

    should_flip = config['should_flip']
    flip_prob = config['flip_prob']
    should_draw = config['should_draw']
    category_map = config['category_map']

    if not os.path.exists(OUTPUT_IMG_DIR): os.makedirs(OUTPUT_IMG_DIR)
    if not os.path.exists(OUTPUT_LABEL_DIR): os.makedirs(OUTPUT_LABEL_DIR)

    img_paths = glob.glob(os.path.join(INPUT_IMG_DIR, '*.png'))

    print(f"找到 {len(img_paths)} 张图片，开始增强...")

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
        transform, AUGMENT_COUNT = build_transforms(augmentation_operations = config['augmentation_operations'],
                                                    bbox_params_config = config['bbox_params'],
                                                    a_count= config['augment_count'],
                                                    bboexes = bboxes[:-1],
                                                    labels = class_labels[:-1],
                                                    category_map = category_map,
                                                    auto = config['auto'])

        # 生成增强图
        for i in range(AUGMENT_COUNT):
            try:
                oorig_box = bboxes  # 不这么写transformed会把bboxes也更新报错
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels, indexes=indexes)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
                aug_indexes = transformed['indexes']  # 获取增强后的索引信息

                # 获取前后的参考框的坐标
                x_c, y_c, w, h = shape_size(aug_bboxes[-1], image.shape)
                orig_x_c, orig_y_c, orig_w, orig_h = shape_size(oorig_box[-1], image.shape)
                orig_corners = calculate_corners(orig_x_c, orig_y_c, orig_w, orig_h)
                aug_corners = calculate_corners(x_c, y_c, w, h)

                # 计算变换矩阵
                H = cv2.getPerspectiveTransform(
                    np.float32(orig_corners),
                    np.float32(aug_corners)
                )

                aug_bboxes = aug_bboxes[:-1]
                aug_labels = aug_labels[:-1]
                aug_indexes = aug_indexes[:-1]

                aug_keypoints = clean_by_index(aug_indexes, keypoints)
                oorig_box = clean_by_index(aug_indexes, oorig_box)  # 对标消去不存在目标框后的各个参数

                if len(aug_bboxes) == 0: continue
                # 根据索引获取变换后存在的目标框，并更新其关键点
                augmented_keypoints = []
                # 处理每一个框和关键点
                for aug_bbox, orig_bbox, kp in zip(aug_bboxes, oorig_box, aug_keypoints):
                    # Step 1: 逆归一化
                    x_c, y_c, w, h = shape_size(aug_bbox, image.shape)
                    orig_x_c, orig_y_c, orig_w, orig_h = shape_size(orig_bbox, image.shape)

                    # Step 2: 计算框的四个角
                    orig_corners = calculate_corners(orig_x_c, orig_y_c, orig_w, orig_h)
                    aug_corners = calculate_corners(x_c, y_c, w, h)

                    updated_kps = []
                    for (x, y, k_cls) in kp:
                        # Step 3: 逆归一化关键点坐标
                        x = np.float64(x) * image.shape[1]  # 逆归一化 X 坐标
                        y = np.float64(y) * image.shape[0]  # 逆归一化 Y 坐标

                        keypoint = np.array([x, y, 1])

                        # 如果坐标为 (0, 0)，表示关键点不可见，跳过
                        if x == 0 and y == 0:
                            new_x = np.float64(0)
                            new_y = np.float64(0)
                        else:
                            # Step 4: 使用四个角来定位新的关键点坐标
                            # 计算仿射矩阵或比例关系进行变换
                            orig_corners = np.float64(orig_corners)
                            aug_corners = np.float64(aug_corners)

                            aug_keypoint = H @ keypoint
                            new_x = aug_keypoint[0] / aug_keypoint[2]
                            new_y = aug_keypoint[1] / aug_keypoint[2]

                        # Step 5: 归一化新的关键点坐标
                        if check_in(new_x, new_y, image.shape[1], image.shape[0]):
                            new_x /= image.shape[1]
                            new_y /= image.shape[0]
                        else:  # 关键点超出范围了
                            new_x = np.float64(0)
                            new_y = np.float64(0)

                        # 保存更新后的关键点
                        if is_gray(aug_img,new_x,new_y):
                            updated_kps.append((0, 0, k_cls))
                        else:
                            updated_kps.append((new_x, new_y, k_cls))
                        #updated_kps.append((new_x, new_y, k_cls))

                    augmented_keypoints.append(updated_kps)

                aug_img, aug_bboxes, augmented_keypoints = flip_image(aug_img, aug_bboxes, augmented_keypoints,
                                                                      should_flip, flip_prob)

                # 保存增强后的图像
                save_name = f"{base_name}_aug_{i}"

                if should_draw:
                    aug_img = image_draw(
                        aug_img,
                        aug_bboxes,
                        augmented_keypoints
                    )

                cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, save_name + '.png'), aug_img)

                # 保存增强后的标签
                with open(os.path.join(OUTPUT_LABEL_DIR, save_name + '.txt'), 'w') as f:
                    for cls, bbox, kp in zip(aug_labels, aug_bboxes, augmented_keypoints):
                        x_c, y_c, w, h = clean_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
                        line = f"{int(cls)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

                        # 添加关键点到标签行
                        for (key_x, key_y, key_cls) in kp:
                            line += f" {key_x:.6f} {key_y:.6f} {int(key_cls)}"

                        f.write(line + ' \n')

            except Exception as e:
                print(f"处理 {base_name} 增强时出错: {e}")

    print("增强完成！")


if __name__ == "__main__":
    augment_data(r"F:\frr\try\utils\datasets_preprocess\image_agument\config\rf.yaml")