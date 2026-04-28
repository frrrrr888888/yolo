from ultralytics.models.yolo.detect import DetectionTrainer

if __name__ == '__main__':
    args = dict(model=r"./ultralytics/cfg/models/11/yolo11.yaml",
                data=r"./ultralytics/cfg/datasets_weed/weed_RF_c2.yaml", epochs=600)
    trainer = DetectionTrainer(cfg=r"./ultralytics/cfg/default_detect.yaml",
                               overrides=args)
    trainer.train()
