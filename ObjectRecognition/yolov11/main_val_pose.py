from ultralytics.models.yolo.pose.val import PoseValidator

if __name__ == '__main__':
    args = dict(
        model=r"",
        data=r".\ultralytics\cfg\datasets_weed\weed_coaxial.yaml")
    validator = PoseValidator(cfg_path=r".\ultralytics\cfg\default_pose_val.yaml",args=args)
    validator()