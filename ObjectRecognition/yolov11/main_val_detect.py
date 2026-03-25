from ultralytics.models.yolo.detect import DetectionValidator

if __name__ == '__main__':
    args = dict(
        model=r"",
                data=r".\ultralytics\cfg\datasets\coco8.yaml")
    validator = DetectionValidator(cfg_path=r".\ultralytics\cfg\default_detect_val.yaml",args=args)
    validator()
