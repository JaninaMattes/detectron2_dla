# import some common libraries
import numpy as np
import os, json, cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import SimpleTrainer
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances


class FineTuner(SimpleTrainer):
    """
    The "SimpleTrainer" of Detectron2 framework to inherit a number 
    of pre-defined logic for a standard training workflow.
    """

    def register_document_dicts_coco(name_train, name_val, json_train_file, json_val_file, image_root, metadata={}):
        """
        For the parsing and mapping of datasets which are in COCO format
        according to the Detectron2 documentation. 
        Register a dataset in COCO's json annotation format for
        instance detection, instance segmentation.
        (i.e., Type 1 in http://cocodataset.org/#format-data.
        `instances*.json`)

        Example Train: "my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir"
        Example Val: "my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir"
        """

        register_coco_instances(name=name_train, metadata=metadata, json_file=json_train_file, image_root=image_root)
        register_coco_instances(name=name_val, metadata=metadata, json_file=json_val_file, image_root=image_root)

    def get_document_dicts(img_dir):
        """
        For the parsing and mapping of datasets which are not in COCO format.
        """
        json_file = os.path.join(img_dir, "via_region_data.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for idx, v in enumerate(imgs_anns.values()):
            record = {}
            
            filename = os.path.join(img_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            annos = v["regions"]
            objs = []
            for _, anno in annos.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    for d in ["train", "val"]:
        DatasetCatalog.register("documents_" + d, lambda d=d: get_document_dicts("documents/" + d))
        MetadataCatalog.get("documents_" + d).set(thing_classes=["documents"])
    documents_metadata = MetadataCatalog.get("documents_train")