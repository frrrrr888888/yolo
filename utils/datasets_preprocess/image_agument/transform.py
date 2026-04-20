import albumentations as A

def build_transforms(augmentation_operations, bbox_params_config):

    # 构造变换操作
    augmentations = []

    if 'Affine' in augmentation_operations:
        augmentations.append(A.Affine(shear=augmentation_operations['Affine']['shear'],
                                      scale=tuple(augmentation_operations['Affine']['scale']),
                                      translate_percent=tuple(augmentation_operations['Affine']['translate_percent']),
                                      fill = tuple(augmentation_operations['Affine']['fill']),
                                      p=augmentation_operations['Affine']['p']))

    if 'Perspective' in augmentation_operations:
        augmentations.append(A.Perspective(scale=tuple(augmentation_operations['Perspective']['scale']),
                                           p=augmentation_operations['Perspective']['p']))

    if 'HorizontalFlip' in augmentation_operations:
        augmentations.append(A.HorizontalFlip(p=augmentation_operations['HorizontalFlip']['p']))
    if 'VerticalFlip' in augmentation_operations:
        augmentations.append(A.VerticalFlip(p=augmentation_operations['VerticalFlip']['p']))
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

    return transform