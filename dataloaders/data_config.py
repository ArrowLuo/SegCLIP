import os.path

DATA_CONFIG_DICT = {}

ROOT_PATH_ = "./"

# coco_train2014.pkl
# coco_train2014_seg_scale224_sigma9.0_min_size224.lmdb
# karpathy
DATA_CONFIG_DICT["coco"] = {
    "train": {
        "features_path": os.path.join(ROOT_PATH_, ""),
        "data_path": os.path.join(ROOT_PATH_, "karpathy"),
    },
    "val": {
        "features_path": os.path.join(ROOT_PATH_, ""),
        "data_path": os.path.join(ROOT_PATH_, "karpathy"),
    },
    "test": None
}

# cc3m_train_desc.pkl
# cc3m_train_lmdb_total
# cc3m_train_lmdb_total_keys.pkl
# cc3m_train_lmdb_total_seg_scale224_sigma9.0_min_size224.lmdb
# Train_GCC-training.tsv
# Validation_GCC-1.1.0-Validation.tsv
DATA_CONFIG_DICT["cc"] = {
    "train": {
        "features_path": os.path.join(ROOT_PATH_, ""),
        "data_path": os.path.join(ROOT_PATH_, ""),
    },
    "val": {
        "features_path": os.path.join(ROOT_PATH_, ""),
        "data_path": os.path.join(ROOT_PATH_, ""),
    },
    "test": None
}

