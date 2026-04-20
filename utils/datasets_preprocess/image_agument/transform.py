import albumentations as A

def decide_augmentation(bboxes, labels):
    from collections import Counter

    if len(bboxes) == 0:
        return ["basic"], 1

    areas = [w*h for (_,_,w,h) in bboxes]

    small_ratio = sum(a < 0.01 for a in areas) / len(areas)
    tiny_ratio  = sum(a < 0.005 for a in areas) / len(areas)

    cnt = Counter(labels)
    weed_ratio = cnt.get('2', 0) / len(labels)
    negweed_ratio = cnt.get('1', 0) / len(labels)

    num_objects = len(bboxes)

    edge_ratio = sum(
        (x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9)
        for (x,y,_,_) in bboxes
    ) / len(bboxes)

    # === 策略标签 ===
    strategy = []

    if tiny_ratio > 0.3:
        strategy.append("strong_scale")

    elif small_ratio > 0.5:
        strategy.append("scale")

    if negweed_ratio > 0.3:
        strategy.append("color")

    if weed_ratio > 0.5:
        strategy.append("noise")

    if num_objects > 10:
        strategy.append("dense")

    if edge_ratio > 0.3:
        strategy.append("shift")

    # === 难度评分 ===
    score = (
        small_ratio*2 +
        tiny_ratio*3 +
        negweed_ratio*2 +
        weed_ratio +
        num_objects/10 +
        edge_ratio*1.5
    )

    n_aug = min(5, max(1, int(score)))

    return strategy, n_aug


def build_transforms(augmentation_operations, bbox_params_config, a_count, bboexes, labels, auto = False):

    # 构造变换操作
    augmentations = []
    if auto:
        print("自动增强")
        transform_list, a_count = decide_augmentation(bboexes, labels)
        # === 几何（必有）===
        if "strong_scale" in transform_list:
            augmentations.append(A.Affine(scale=(1.0, 1.6),fill = tuple(augmentation_operations['Affine']['fill']), p=0.8))

        elif "scale" in transform_list:
            augmentations.append(A.Affine(scale=(0.8, 1.3),fill = tuple(augmentation_operations['Affine']['fill']), p=0.7))

        else:
            augmentations.append(A.Affine(scale=(0.9, 1.1),fill = tuple(augmentation_operations['Affine']['fill']), p=0.5))

        # === 位移 ===
        if "shift" in transform_list:
            augmentations.append(A.Affine(translate_percent=(0.1, 0.1),fill = tuple(augmentation_operations['Affine']['fill']) ,p=0.5))

        # === 密集 → crop ===
        if "dense" in transform_list:
            augmentations.append(A.RandomCrop(height=640, width=640,fill = tuple(augmentation_operations['Affine']['fill']), p=1))


        # === 颜色增强（OneOf）===
        if "color" in transform_list:
            augmentations.append(
                A.OneOf([
                    #A.HueSaturationValue(...),
                    A.HueSaturationValue(
                        hue_shift_limit=augmentation_operations['HueSaturationValue']['hue_shift_limit'],
                        sat_shift_limit=augmentation_operations['HueSaturationValue']['sat_shift_limit'],
                        val_shift_limit=augmentation_operations['HueSaturationValue']['val_shift_limit']),
                    #A.RandomBrightnessContrast(...)
                    A.RandomBrightnessContrast(
                        brightness_limit=augmentation_operations['RandomBrightnessContrast']['brightness_limit'],
                        contrast_limit=augmentation_operations['RandomBrightnessContrast']['contrast_limit'])
                ], p=0.6)
            )

        # === 噪声增强 ===
        if "noise" in transform_list:
            augmentations.append(
                A.OneOf([
                    #A.GaussNoise(...),
                    A.GaussNoise(std_range=augmentation_operations['GaussNoise']['std_range']),
                    #A.MotionBlur(...)
                    A.MotionBlur(blur_limit=augmentation_operations['MotionBlur']['blur_limit'])
                ], p=0.4)
            )

        # === 遮挡（通用）===
        #augmentations.append(A.CoarseDropout(..., p=0.3))
        if 'CoarseDropout' in augmentation_operations:
            augmentations.append(A.CoarseDropout(
                num_holes_range=augmentation_operations['CoarseDropout']['num_holes_range'],
                hole_height_range=augmentation_operations['CoarseDropout']['hole_height_range'],
                hole_width_range=augmentation_operations['CoarseDropout']['hole_width_range'],
                fill=tuple(augmentation_operations['CoarseDropout']['fill']),
                p=augmentation_operations['CoarseDropout']['p']
            ))

        transform = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(
                format=bbox_params_config['format'],  # 从配置文件加载格式
                min_visibility=bbox_params_config['min_visibility'],  # 最小可见度
                label_fields=bbox_params_config['label_fields']
            )
        )

        return transform, a_count
    else:
        if 'Affine' in augmentation_operations:
            augmentations.append(A.Affine(shear=augmentation_operations['Affine']['shear'],
                                          scale=tuple(augmentation_operations['Affine']['scale']),
                                          translate_percent=tuple(augmentation_operations['Affine']['translate_percent']),
                                          fill = tuple(augmentation_operations['Affine']['fill']),
                                          p=augmentation_operations['Affine']['p']))

        if 'Perspective' in augmentation_operations:
            augmentations.append(A.Perspective(scale=tuple(augmentation_operations['Perspective']['scale']),
                                               p=augmentation_operations['Perspective']['p']))

        if 'RandomBrightnessContrast' in augmentation_operations:
            augmentations.append(A.RandomBrightnessContrast(
                brightness_limit=augmentation_operations['RandomBrightnessContrast']['brightness_limit'],
                contrast_limit=augmentation_operations['RandomBrightnessContrast']['contrast_limit'],
                p=augmentation_operations['RandomBrightnessContrast']['p']
            ))
        if 'HueSaturationValue' in augmentation_operations:
            augmentations.append(A.HueSaturationValue(
                hue_shift_limit=augmentation_operations['HueSaturationValue']['hue_shift_limit'],
                sat_shift_limit=augmentation_operations['HueSaturationValue']['sat_shift_limit'],
                val_shift_limit=augmentation_operations['HueSaturationValue']['val_shift_limit'],
                p=augmentation_operations['HueSaturationValue']['p']
            ))
        if 'GaussNoise' in augmentation_operations:
            augmentations.append(A.GaussNoise(
                std_range=augmentation_operations['GaussNoise']['std_range'],
                p=augmentation_operations['GaussNoise']['p']
            ))
        if 'MotionBlur' in augmentation_operations:
            augmentations.append(A.MotionBlur(
                blur_limit=augmentation_operations['MotionBlur']['blur_limit'],
                p=augmentation_operations['MotionBlur']['p']
            ))
        if 'CoarseDropout' in augmentation_operations:
            augmentations.append(A.CoarseDropout(
                num_holes_range=augmentation_operations['CoarseDropout']['num_holes_range'],
                hole_height_range=augmentation_operations['CoarseDropout']['hole_height_range'],
                hole_width_range=augmentation_operations['CoarseDropout']['hole_width_range'],
                fill=tuple(augmentation_operations['CoarseDropout']['fill']),
                p=augmentation_operations['CoarseDropout']['p']
            ))

        # 合并成一个变换操作
        transform = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(
                format=bbox_params_config['format'],  # 从配置文件加载格式
                min_visibility=bbox_params_config['min_visibility'],  # 最小可见度
                label_fields=bbox_params_config['label_fields']
            )
        )

        return transform, a_count