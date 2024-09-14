# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

DatasetCatalog.clear()
MetadataCatalog.clear()


# ==== Predefined datasets and splits for COCO ==========
# Dataset
CLASS_NAMES = ["_background_","Table","Text","Title","Image"]
# Dataset path
DATASET_ROOT = '/path/to/TCA'
# Path to the annotations
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'Train')
VAL_PATH = os.path.join(DATASET_ROOT, 'val')

TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')

# Subset of the dataset
PREDEFINED_SPLITS_DATASET = {
    "TCA_train": (TRAIN_PATH, TRAIN_JSON),
    "TCA_val": (VAL_PATH, VAL_JSON),
}

def register_all_TCA(root):
    # train
    DatasetCatalog.register("TCA_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("TCA_train").set(thing_classes=CLASS_NAMES,
                                          evaluator_type='coco',
                                          json_file=TRAIN_JSON,
                                          image_root=TRAIN_PATH)


    DatasetCatalog.register("TCA_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("TCA_val").set(thing_classes=CLASS_NAMES,
                                        evaluator_type='coco',
                                        json_file=VAL_JSON,
                                        image_root=VAL_PATH)




# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_TCA(_root)
