import albumentations as A
from collections import Counter
import numpy as np

def decide_augmentation(bboxes, labels, category_map):

    if len(bboxes) == 0:
        return ["basic"], 1

    # ===== 构建 label → category 映射 =====
    label2cat = {}
    for k, v in category_map.items():
        for label in v:
            label2cat[int(label)] = k  # 保证是int

    # ===== 类别映射 =====
    categories = [label2cat.get(int(l), "unknown") for l in labels]
    cnt = Counter(categories)

    total = len(labels)

    weed_ratio = cnt.get("weed", 0) / total
    crop_ratio = cnt.get("crop", 0) / total
    negweed_ratio = cnt.get("negweed", 0) / total

    # ===== 面积 =====
    areas = [w*h for (_,_,w,h) in bboxes]
    small_ratio = sum(a < 0.01 for a in areas) / total
    tiny_ratio  = sum(a < 0.005 for a in areas) / total

    # ===== 密度 =====
    num_objects = len(bboxes)

    # ===== 边缘 =====
    edge_ratio = sum(
        (x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9)
        for (x,y,_,_) in bboxes
    ) / total

    # ===== 距离 =====
    def center(box):
        x, y, w, h = box
        return np.array([x + w/2, y + h/2])

    crop_centers = [
        center(bboxes[i]) for i in range(len(labels))
        if categories[i] == "crop"
    ]

    neg_centers = [
        center(bboxes[i]) for i in range(len(labels))
        if categories[i] == "negweed"
    ]

    min_dist = None
    if crop_centers and neg_centers:
        min_dist = min(
            np.linalg.norm(c - n)
            for c in crop_centers
            for n in neg_centers
        )

    # ===== 策略 =====
    strategy = []

    # 1. hardest case
    if min_dist is not None and min_dist < 0.1:
        strategy.append("hard_neg")
    elif negweed_ratio > 0.3:
        strategy.append("color")

    # 2. small object
    if tiny_ratio > 0.3:
        strategy.append("strong_scale")
    elif small_ratio > 0.5:
        strategy.append("scale")

    # 3. weed多
    if weed_ratio > 0.5:
        strategy.append("noise")

    # 4. 密集
    if num_objects > 10:
        strategy.append("dense")

    # 5. 边缘
    if edge_ratio > 0.3:
        strategy.append("shift")

    # 6. crop保护
    if crop_ratio > 0.4:
        strategy.append("protect_crop")

    # ===== 难度评分 =====
    dist_score = 0
    if min_dist is not None:
        dist_score = 1.0 / (min_dist + 1e-3)

    score = (
        small_ratio * 2 +
        tiny_ratio * 3 +
        negweed_ratio * 3 +
        weed_ratio +
        num_objects / 10 +
        edge_ratio * 1.5 +
        dist_score * 2
    )

    n_aug = min(5, max(1, int(score)))

    return strategy, n_aug


def build_transforms(augmentation_operations, bbox_params_config, a_count, bboexes, labels, category_map, auto = False):

    # 构造变换操作
    augmentations = []
    if auto:
        print("自动增强")

        transform_list, a_count = decide_augmentation(bboexes, labels,category_map)

        augmentations = []
        fill_color = tuple(augmentation_operations.Affine.fill)

        # ===== 1. 几何（核心增强，受策略控制）=====
        # 🔥 crop保护（关键）
        if "protect_crop" in transform_list:
            scale_aug = A.Affine(scale=(0.95, 1.05), fill=fill_color, p=0.5)

        elif "strong_scale" in transform_list:
            scale_aug = A.Affine(scale=(1.0, 1.6), fill=fill_color, p=0.8)

        elif "scale" in transform_list:
            scale_aug = A.Affine(scale=(0.8, 1.3), fill=fill_color, p=0.7)

        else:
            scale_aug = A.Affine(scale=(0.9, 1.1), fill=fill_color, p=0.5)

        augmentations.append(scale_aug)

        # ===== 2. 位移 =====
        if "shift" in transform_list:
            augmentations.append(
                A.Affine(translate_percent=(0.1, 0.1), fill=fill_color, p=0.5)
            )

        # ===== 3. 密集场景 → 裁剪 =====
        """if "dense" in transform_list:
            augmentations.append(
                A.RandomCrop(height=640, width=640, p=1)
            )
        else:
            augmentations.append(
                A.RandomCrop(height=640, width=640, p=1)
            )
        augmentations.append(
            A.RandomCrop(height=640, width=640, p=1)
        )"""

        # ===== 4. HARD NEG（新增核心）=====
        if "hard_neg" in transform_list:
            # 限制颜色增强（防止破坏细节）
            color_p = 0.3
        else:
            color_p = 0.6

        # ===== 5. 颜色增强 =====
        if "color" in transform_list:
            '''augmentations.append(
                A.OneOf([
                    A.HueSaturationValue(
                        hue_shift_limit=augmentation_operations['HueSaturationValue']['hue_shift_limit'],
                        sat_shift_limit=augmentation_operations['HueSaturationValue']['sat_shift_limit'],
                        val_shift_limit=augmentation_operations['HueSaturationValue']['val_shift_limit']
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=augmentation_operations['RandomBrightnessContrast']['brightness_limit'],
                        contrast_limit=augmentation_operations['RandomBrightnessContrast']['contrast_limit']
                    )
                ], p=color_p)
            )'''
            augmentations.append(A.HueSaturationValue(
                hue_shift_limit=augmentation_operations.HueSaturationValue.hue_shift_limit,
                sat_shift_limit=augmentation_operations.HueSaturationValue.sat_shift_limit,
                val_shift_limit=augmentation_operations.HueSaturationValue.val_shift_limit,
                p=color_p
            ))
            augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=augmentation_operations.RandomBrightnessContrast.brightness_limit,
                contrast_limit=augmentation_operations.RandomBrightnessContrast.contrast_limit,
                p=color_p
            ))

        # ===== 6. 噪声（弱化在保护场景）=====
        if "noise" in transform_list:
            noise_p = 0.4

            if "protect_crop" in transform_list:
                noise_p = 0.2  # 🔥 降低扰动

            augmentations.append(A.GaussNoise(
                std_range=augmentation_operations.GaussNoise.std_range,
                p=noise_p
            ))
            augmentations.append(A.MotionBlur(
                blur_limit=augmentation_operations.MotionBlur.blur_limit,
                p=noise_p
            ))

        # ===== 7. 遮挡 =====
        if 'CoarseDropout' in augmentation_operations:

            dropout_p = augmentation_operations.CoarseDropout.p

            # 🔥 HARD NEG时减少遮挡（保留细节）
            if "hard_neg" in transform_list:
                dropout_p *= 0.5

            augmentations.append(
                A.CoarseDropout(
                    num_holes_range=augmentation_operations.CoarseDropout.num_holes_range,
                    hole_height_range=augmentation_operations.CoarseDropout.hole_height_range,
                    hole_width_range=augmentation_operations.CoarseDropout.hole_width_range,
                    fill=tuple(augmentation_operations.CoarseDropout.fill),
                    p=dropout_p
                )
            )


        # ===== Compose =====
        transform = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(
                format=bbox_params_config.format,
                min_visibility=bbox_params_config.min_visibility,
                label_fields=bbox_params_config.abel_fields
            )
        )
        print("增加张数：",a_count)
        return transform, a_count
    else:
        if 'Affine' in augmentation_operations:
            augmentations.append(A.Affine(shear=augmentation_operations.Affine.shear,
                                          scale=tuple(augmentation_operations.Affine.scale),
                                          translate_percent=tuple(augmentation_operations.Affine.translate_percent),
                                          fill = tuple(augmentation_operations.Affine.fill),
                                          p=augmentation_operations.Affine.p))

        if 'Perspective' in augmentation_operations:
            augmentations.append(A.Perspective(scale=tuple(augmentation_operations.Perspective.scale),
                                               p=augmentation_operations.Perspective.p))

        if 'RandomBrightnessContrast' in augmentation_operations:
            augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=augmentation_operations.RandomBrightnessContrast.brightness_limit,
                contrast_limit=augmentation_operations.RandomBrightnessContrast.contrast_limit,
                p=augmentation_operations.RandomBrightnessContrast.p
            ))
        if 'HueSaturationValue' in augmentation_operations:
            augmentations.append(A.HueSaturationValue(
                hue_shift_limit=augmentation_operations.HueSaturationValue.hue_shift_limit,
                sat_shift_limit=augmentation_operations.HueSaturationValue.sat_shift_limit,
                val_shift_limit=augmentation_operations.HueSaturationValue.val_shift_limit,
                p=augmentation_operations.HueSaturationValue.p
            ))
        if 'GaussNoise' in augmentation_operations:
            augmentations.append(A.GaussNoise(
                std_range=augmentation_operations.GaussNoise.std_range,
                p=augmentation_operations.GaussNoise.p
            ))
        if 'MotionBlur' in augmentation_operations:
            augmentations.append(A.MotionBlur(
                blur_limit=augmentation_operations.MotionBlur.blur_limit,
                p=augmentation_operations.MotionBlur.p
            ))
        if 'CoarseDropout' in augmentation_operations:
            augmentations.append(A.CoarseDropout(
                num_holes_range=augmentation_operations.CoarseDropout.num_holes_range,
                hole_height_range=augmentation_operations.CoarseDropout.hole_height_range,
                hole_width_range=augmentation_operations.CoarseDropout.hole_width_range,
                fill=tuple(augmentation_operations.CoarseDropout.fill),
                p=augmentation_operations.CoarseDropout.p
            ))

        # 合并成一个变换操作
        transform = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(
                format=bbox_params_config.format,  # 从配置文件加载格式
                min_visibility=bbox_params_config.min_visibility,  # 最小可见度
                label_fields=bbox_params_config.label_fields
            )
        )

        return transform, a_count