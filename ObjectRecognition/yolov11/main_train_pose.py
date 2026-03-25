from ultralytics.models.yolo.pose import PoseTrainer

if __name__ == '__main__':
    args = dict(
        model=r".\ultralytics\cfg\models\11\yolo11-pose.yaml",
                data=r"D.\ultralytics\cfg\datasets\weed_coaxial.yaml",
                epochs=300)
    trainer = PoseTrainer(cfg=r"./ultralytics/cfg/default_pose.yaml",overrides=args)
    trainer.train()
