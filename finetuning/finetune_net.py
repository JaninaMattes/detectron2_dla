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
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("detectron2")

class FineTuner(DefaultTrainer):
    """
    The "SimpleTrainer" of Detectron2 framework to inherit a number 
    of pre-defined logic for a standard training workflow.

    Pre-trained model: https://www.dropbox.com/sh/jxuxu2oh4f8ogms/AADaG0U2hXORh_kd8NazDAgsa?dl=0
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

def transfer_pretrained_weights(model, pretrained_model_pth):
    pretrained_weights = torch.load(pretrained_model_pth)['model']
    new_dict = {k.replace('module.',''):v for k, v in pretrained_weights.items()
                if 'cls_score' not in k and 'bbox_pred' not in k}
    this_state = model.state_dict()
    this_state.update(new_dict)
    model.load_state_dict(this_state)
    return model

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze() #freezes the first two layers of the backbone
    default_setup(cfg, args)
    return cfg

def main(args, pretrained_model_pth="models/Resnet-101/model_final.pth"):
    """ source: https://github.com/facebookresearch/maskrcnn-benchmark/issues/15
    """
    cfg = setup(args)

    if args.eval_only:       
    
        model = FineTuner.build_model(cfg)
        model = DetectionCheckpointer(model).load(pretrained_model_pth)
        model = transfer_pretrained_weights(model, pretrained_model_pth) # Todo Test
        
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
    consider writing your own training loop or subclassing the trainer.
    """
    # trainer = FineTuner(cfg)
    FineTuner.resume_or_load(resume=args.resume)

    if cfg.TEST.AUG.ENABLED:
        FineTuner.register_hooks(
            [hooks.EvalHook(0, lambda: FineTuner.test_with_TTA(cfg, FineTuner.model))]
        )
    return FineTuner.train()


if __name__ == "__main__":

    # parse arguments
    args = default_argument_parser().parse_args()

    # output
    output_dir = os.path.join('./output', datetime.datetime.now().strftime('%Y%m%dT%H%M'))
    os.makedirs(output_dir, exist_ok=True)
    
    # logging
    logger = setup_logger(output=output_dir)
    logger.info("Command Line Args:", args)

    # register custom dataset in COCO format
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

    # register metadata
    metadata_train = MetadataCatalog.get("dla_train")
    metadata_val = MetadataCatalog.get("dla_val")

    # create config
    cfg = get_cfg()
    # mask rcnn resnet101
    cfg.merge_from_file("configs/DLA_mask_rcnn_R_101_FPN_3x.yaml")
    
    # mask rcnn resnext
    # cfg.merge_from_file("configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.OUTPUT_DIR = output_dir

    logger.info(cfg)

    # serialize the training config
    cfg_str = cfg.dump()

    with open(os.path.join(cfg.OUTPUT_DIR, "finetuning_config.yaml"), "w") as f:
        f.write(cfg_str)
    f.close()

    trainer = FineTuner(cfg)
    FineTuner.resume_or_load(resume=False)
    FineTuner.train()