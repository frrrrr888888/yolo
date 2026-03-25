# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# 完整 val_no_class.py（PoseValidator），在原有表格输出基础上，新增“类别无关关键点指标”小表，
# 并仅在“有可见关键点的 GT”上评估，同时剔除无关键点类别（默认 cab=0）。

from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """
    Pose 验证器：在检测验证基础上加入关键点度量。
    新增：类别无关关键点指标（KPnc_*），只看关键点是否匹配，不要求类别一致。
    修正：no-class 只统计“有可见关键点的 GT”，并剔除 cab 等无关键点类别的 GT/Pred。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()

        # —— 配置：哪些类别在 no-class 评估中需要排除（如 cab 无关键点） —— #
        # 把 cab 的 id 设为 0；如有更多无关键点类别，也可加进集合
        self._kp_exclude_cls = {0}

        # 累积“类别无关关键点”统计：元素 (tp_p_nc_bool[N_pred, niou], conf[N_pred], n_gt_int)
        self._kp_nc_stats: List[Tuple[np.ndarray, np.ndarray, int]] = []

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def get_desc(self) -> str:
        # 保持原有主表表头
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> Dict[str, torch.Tensor]:
        preds = super().postprocess(preds)
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # remove extra if exists
        return preds

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        pbatch["keypoints"] = kpts
        return pbatch

    def _prepare_pred(self, pred: Dict[str, Any], pbatch: Dict[str, Any]) -> Dict[str, Any]:
        predn = super()._prepare_pred(pred, pbatch)
        predn["keypoints"] = ops.scale_coords(
            pbatch["imgsz"], pred.get("keypoints").clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )
        return predn

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        - 原有 tp：框匹配（基类）
        - 原有 tp_p：类别有关关键点匹配
        - 新增 tp_p_nc：类别无关关键点匹配（把 pred/gt 类别都视作同一类）【但仅在“有可见关键点的 GT”上评估，且剔除 cab 等类别】
        """
        tp = super()._process_batch(preds, batch)

        # —— 原有（类别有关）关键点匹配，保持不变 —— #
        gt_cls_all = batch["cls"]
        n_pred_all = int(len(preds["cls"]))
        if len(gt_cls_all) == 0 or n_pred_all == 0:
            tp_p = np.zeros((n_pred_all, self.niou), dtype=bool)
        else:
            area_all = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou_all = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area_all)
            tp_p = self.match_predictions(preds["cls"], gt_cls_all, iou_all).cpu().numpy()
        tp.update({"tp_p": tp_p})

        # —— 类别无关关键点匹配（修正：只用“有可见关键点”的 GT，并剔除无关键点类别） —— #
        if n_pred_all == 0:
            tp_p_nc = np.zeros((0, self.niou), dtype=bool)
            tp.update({"tp_p_nc": tp_p_nc})
            return tp

        # 1) 选择“有效 GT”：至少有一个关键点可见，且类别不在排除集合
        gt_kpts = batch["keypoints"]           # (N_gt, K, 3)
        gt_vis = gt_kpts[..., 2] > 0           # (N_gt, K)
        gt_has_any_kpt = gt_vis.any(dim=1)     # (N_gt,)
        gt_cls_vec = gt_cls_all.long().view(-1)
        if len(self._kp_exclude_cls) > 0:
            excl = torch.tensor(list(self._kp_exclude_cls), device=gt_cls_vec.device, dtype=gt_cls_vec.dtype)
            gt_keep = gt_has_any_kpt & (~torch.isin(gt_cls_vec, excl))
        else:
            gt_keep = gt_has_any_kpt

        # 2) 选择“有效 Pred”：类别不在排除集合
        pred_cls_vec = preds["cls"].long().view(-1)
        if len(self._kp_exclude_cls) > 0:
            excl_p = torch.tensor(list(self._kp_exclude_cls), device=pred_cls_vec.device, dtype=pred_cls_vec.dtype)
            pred_keep = ~torch.isin(pred_cls_vec, excl_p)
        else:
            pred_keep = torch.ones_like(pred_cls_vec, dtype=torch.bool)

        if gt_keep.sum() == 0 or pred_keep.sum() == 0:
            # 没有可用的 GT 或 Pred，则本图 no-class 统计为空
            tp_p_nc_valid = np.zeros((int(pred_keep.sum().item()), self.niou), dtype=bool)
            conf_valid = preds["conf"][pred_keep].detach().cpu().numpy().astype(np.float32)
            n_gt_valid = int(gt_keep.sum().item())
            self._kp_nc_stats.append((tp_p_nc_valid, conf_valid, n_gt_valid))
            # 同时给 tp 字典放一个“对齐原 N_pred 的占位”（不参与主流程，仅便于调试）
            tp_p_nc = np.zeros((n_pred_all, self.niou), dtype=bool)
            tp.update({"tp_p_nc": tp_p_nc})
            return tp

        # 3) 计算有效子集上的 kpt_iou 并进行“类别无关”匹配
        area_valid = ops.xyxy2xywh(batch["bboxes"][gt_keep])[:, 2:].prod(1) * 0.53
        iou_valid = kpt_iou(gt_kpts[gt_keep], preds["keypoints"][pred_keep], sigma=self.sigma, area=area_valid)

        # ★ 修正点：zeros_pred / zeros_gt 必须是一维 (N,) 才符合 match_predictions 的期望
        n_pred_valid = int(pred_keep.sum().item())
        n_gt_valid = int(gt_keep.sum().item())
        zeros_pred = torch.zeros((n_pred_valid,), device=preds["cls"].device, dtype=preds["cls"].dtype)
        zeros_gt = torch.zeros((n_gt_valid,), device=gt_cls_all.device, dtype=gt_cls_all.dtype)

        tp_p_nc_valid = self.match_predictions(zeros_pred, zeros_gt, iou_valid).cpu().numpy()

        conf_valid = preds["conf"][pred_keep].detach().cpu().numpy().astype(np.float32)

        # —— 累积到全局 no-class 统计 —— #
        self._kp_nc_stats.append((tp_p_nc_valid.astype(bool), conf_valid, n_gt_valid))

        # 为了对齐形状，给 tp 字典补一个“全长”的占位（不用于主流程，仅便于调试）
        tp_p_nc = np.zeros((n_pred_all, self.niou), dtype=bool)
        # 把 valid preds 的 tp 写回占位矩阵对应的位置（可选）
        tp_p_nc[np.where(pred_keep.cpu().numpy())[0]] = tp_p_nc_valid
        tp.update({"tp_p_nc": tp_p_nc})

        return tp

    # ====== 计算“类别无关关键点”指标 ====== #
    def _compute_class_agnostic_kp(self) -> Dict[str, float]:
        """
        使用 self._kp_nc_stats 计算：
          - KPnc-AP50（IoU=0.50 的 AP）
          - KPnc-mAP50-95（10 阈值均值）
          - KPnc-P@50 / KPnc-R@50（IoU=0.50 时，所有预测整体累计的最终精度/召回）
        只基于“有可见关键点”且类别不在排除集的 GT 与 Pred。
        """
        if len(self._kp_nc_stats) == 0:
            return {"kp_nc_ap50": 0.0, "kp_nc_ap": 0.0, "kp_nc_p": 0.0, "kp_nc_r": 0.0}

        tp_list, conf_list, n_gt_list = zip(*self._kp_nc_stats)
        if len(tp_list) == 0:
            return {"kp_nc_ap50": 0.0, "kp_nc_ap": 0.0, "kp_nc_p": 0.0, "kp_nc_r": 0.0}

        tp = np.concatenate(tp_list, axis=0).astype(bool)            # (N_pred_valid, niou)
        conf = np.concatenate(conf_list, axis=0).astype(np.float32)  # (N_pred_valid,)
        n_gt_total = int(np.sum(np.asarray(n_gt_list, dtype=np.int64)))

        if tp.shape[0] == 0 or n_gt_total == 0:
            return {"kp_nc_ap50": 0.0, "kp_nc_ap": 0.0, "kp_nc_p": 0.0, "kp_nc_r": 0.0}

        # 置信度降序
        order = np.argsort(-conf)
        tp = tp[order]

        ap_per_iou = []
        # 便于直观：IoU=0.50 上的整体 Precision/Recall（所有预测累计）
        tpi_50 = tp[:, 0].astype(np.int32)
        cum_tp_50 = np.cumsum(tpi_50)
        cum_fp_50 = np.cumsum(1 - tpi_50)
        final_tp_50 = int(cum_tp_50[-1]) if len(cum_tp_50) else 0
        final_fp_50 = int(cum_fp_50[-1]) if len(cum_fp_50) else 0
        kp_nc_p = (final_tp_50 / max(final_tp_50 + final_fp_50, 1)) if (final_tp_50 + final_fp_50) > 0 else 0.0
        kp_nc_r = (final_tp_50 / max(n_gt_total, 1)) if n_gt_total > 0 else 0.0

        # 逐 IoU 阈值计算 101 点插值 AP
        for i in range(self.niou):
            tpi = tp[:, i].astype(np.int32)
            fpi = 1 - tpi
            cum_tp = np.cumsum(tpi)
            cum_fp = np.cumsum(fpi)

            recall = cum_tp / max(n_gt_total, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

            r_points = np.linspace(0, 1, 101)
            last_p = float(precision[-1]) if precision.size > 0 else 0.0
            pr_interp = np.interp(r_points, recall, precision, left=0.0, right=last_p)
            ap_i = float(np.mean(pr_interp))
            ap_per_iou.append(ap_i)

        ap_per_iou = np.asarray(ap_per_iou, dtype=np.float32)
        kp_nc_ap50 = float(ap_per_iou[0])      # IoU=0.50
        kp_nc_map = float(np.mean(ap_per_iou)) # IoU=0.50:0.95

        return {
            "kp_nc_ap50": kp_nc_ap50,
            "kp_nc_ap": kp_nc_map,
            "kp_nc_p": float(kp_nc_p),
            "kp_nc_r": float(kp_nc_r),
        }

    # ====== 打印：先调用父类打印原表，再打印“无类别关键点”小表 ====== #
    def print_results(self) -> None:
        # 打印原有主表
        super().print_results()

        # 计算并打印修正后的 KPnc 小表
        kp_nc = self._compute_class_agnostic_kp()

        header = ("%22s" + "%11s" * 4) % (
            "No-class KP (all)",
            "AP50",
            "mAP50-95",
            "P@50",
            "R@50",
        )
        row = ("%22s" + "%11.3f" * 4) % (
            "all",
            kp_nc["kp_nc_ap50"],
            kp_nc["kp_nc_ap"],
            kp_nc["kp_nc_p"],
            kp_nc["kp_nc_r"],
        )

        LOGGER.info(header)
        LOGGER.info(row)

        # 也保存一份文本文件
        try:
            with open(self.save_dir / "no_class_keypoint_metrics.txt", "w") as f:
                f.write(
                    f"kp_nc_ap50: {kp_nc['kp_nc_ap50']:.6f}\n"
                    f"kp_nc_ap: {kp_nc['kp_nc_ap']:.6f}\n"
                    f"kp_nc_p: {kp_nc['kp_nc_p']:.6f}\n"
                    f"kp_nc_r: {kp_nc['kp_nc_r']:.6f}\n"
                )
        except Exception as e:
            LOGGER.warning(f"Save no-class KP metrics failed: {e}")

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        from ultralytics.engine.results import Results
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], filename: str) -> None:
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c, k in zip(
            box.tolist(),
            predn["conf"].tolist(),
            predn["cls"].tolist(),
            predn["keypoints"].flatten(1, 2).tolist(),
        ):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "keypoints": k,
                    "score": round(s, 5),
                }
            )

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        # 保持原有 COCO 风格评估返回（bbox + keypoints），no-class 的打印已在 print_results 完成
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        pred_json = self.save_dir / "predictions.json"  # predictions
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
