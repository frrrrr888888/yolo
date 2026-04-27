from ultralytics.models.yolo.detect import DetectionTrainer

if __name__ == '__main__':
    args = dict(model=r"./ultralytics/cfg/models/26/yolo26.yaml",
                data=r"./ultralytics/cfg/datasets_weed/weed_RF_20260421_c2.yaml", epochs=300)
    trainer = DetectionTrainer(cfg=r"./ultralytics/cfg/default.yaml",
                               overrides=args)
    trainer.train()
