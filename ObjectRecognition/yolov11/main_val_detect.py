from ultralytics.models.yolo.detect import DetectionValidator

if __name__ == '__main__':
    args = dict(
                model=r"F:\frr\try\ObjectRecognition\yolov11\runs\detect\train\weights\last.pt",
                data=r"./ultralytics/cfg/datasets_weed/weed_RF_c2.yaml")
    validator = DetectionValidator(cfg_path=r".\ultralytics\cfg\default_detect_val.yaml",args=args)
    validator()
