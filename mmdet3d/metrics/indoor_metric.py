# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.core import get_box_type, indoor_eval
from mmdet3d.registry import METRICS
from mmdet.core import eval_map


@METRICS.register_module()
class IndoorMetric(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (list[float]): List of iou threshold when calculate the
            metric. Defaults to  [0.25, 0.5].
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    """

    def __init__(self,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super(IndoorMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = iou_thr

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        batch_eval_anns = [
            item['data_sample']['eval_ann_info'] for item in data_batch
        ]
        for eval_ann, pred_dict in zip(batch_eval_anns, predictions):
            pred_3d = pred_dict['pred_instances_3d']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        ann_infos = []
        pred_results = []

        for eval_ann, sinlge_pred_results in results:
            ann_infos.append(eval_ann)
            pred_results.append(sinlge_pred_results)

        box_type_3d, box_mode_3d = get_box_type(
            self.dataset_meta['box_type_3d'])

        ret_dict = indoor_eval(
            ann_infos,
            pred_results,
            self.iou_thr,
            self.dataset_meta['CLASSES'],
            logger=logger,
            box_mode_3d=box_mode_3d)

        return ret_dict


@METRICS.register_module()
class Indoor2DMetric(BaseMetric):
    """indoor 2d predictions evaluation metric.

    Args:
        iou_thr (list[float]): List of iou threshold when calculate the
            metric. Defaults to  [0.5].
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    """

    def __init__(self,
                 iou_thr: List[float] = [0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super(Indoor2DMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = iou_thr

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        batch_eval_anns = [
            item['data_sample']['eval_ann_info'] for item in data_batch
        ]
        for eval_ann, pred_dict in zip(batch_eval_anns, predictions):
            pred = pred_dict['pred_instances']
            ann = dict(
                labels=eval_ann['gt_bboxes_labels'],
                bboxes=eval_ann['gt_bboxes'])

            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(self.dataset_meta['CLASSES'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        annotations, preds = zip(*results)
        eval_results = OrderedDict()
        iou_thr_2d = (self.iou_thr) if isinstance(self.iou_thr,
                                                  float) else self.iou_thr
        for iou_thr_2d_single in iou_thr_2d:
            mean_ap, _ = eval_map(
                preds,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr_2d_single,
                dataset=self.dataset_meta['CLASSES'],
                logger=logger)
            eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
        return eval_results