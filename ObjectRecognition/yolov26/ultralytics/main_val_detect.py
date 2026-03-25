from ultralytics.models.yolo.detect import DetectionValidator

if __name__ == '__main__':
    args = dict(
        model=r"C:\Users\32449\Downloads\yolo26n.pt",
                data=r"./ultralytics/cfg/datasets/coco8.yaml")
    # validator = DetectionValidator(cfg_path=r"./ultralytics/cfg/default.yaml",args=args)
    validator = DetectionValidator(args=args)
    validator()
