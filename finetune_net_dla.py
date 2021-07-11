# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

from detectron2.engine.defaults import DefaultTrainer
import logging
import os
import datetime
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import *
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

# from mlflow import log_metric, log_param, log_artifacts

logger = logging.getLogger("detectron2")


class FineTuner(DefaultTrainer):
    """
    We use the "DefaultFineTuner" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleFineTuner", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=False,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, False, output_folder))
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = FineTuner.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = FineTuner.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(FineTuner.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the FineTuner.
    """
    fineTuner = FineTuner(cfg)
    fineTuner.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        fineTuner.register_hooks(
            [hooks.EvalHook(0, lambda: fineTuner.test_with_TTA(cfg, fineTuner.model))]
        )
    return fineTuner.train()


if __name__ == "__main__":
    """
    Training with single GPU:
    --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
    
    """
    args = default_argument_parser().parse_args()
    output_dir = os.path.join('./output', datetime.datetime.now().strftime('%Y%m%dT%H%M'))
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output=output_dir)
    # logger.info("Command Line Args:", args)

    register_coco_instances(
        "dla_train",
        {},
        "./datasets/siemens/train/train2021.json",
        "./datasets/siemens/train/images"
    )

    register_coco_instances(
        "dla_val",
        {},
        "./datasets/siemens/val/val2021.json",
        "./datasets/siemens/val/images"
    )

    metadata_train = MetadataCatalog.get("dla_train")
    metadata_val = MetadataCatalog.get("dla_val")

    cfg = get_cfg()
    # mask rcnn resnet101
    # cfg.merge_from_file("configs/DLA_mask_rcnn_R_101_FPN_3x.yaml")
    # mask rcnn resnext
    cfg.merge_from_file("configs/DLA_mask_rcnn_R_101_FPN_3x_Finetune.yaml")

    cfg.OUTPUT_DIR = output_dir

    logger.info(cfg)

    # serialize the training config
    cfg_str = cfg.dump()
    with open(os.path.join(cfg.OUTPUT_DIR, "finetune_config.yaml"), "w") as f:
        f.write(cfg_str)
    f.close()

    fineTuner = FineTuner(cfg)
    fineTuner.resume_or_load(resume=False)
    fineTuner.train()

    # log_artifacts("outputs")

