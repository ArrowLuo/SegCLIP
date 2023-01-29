from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import io
import zlib
import numpy as np
import pickle
import json
from collections import defaultdict
import lmdb
import base64
from dataloaders.rawimage_util import RawImageExtractor
from dataloaders.rawimage_util import get_felzenszwalb_from_cache
from dataloaders.dataloader_base import DatasetBase

class GCC_DataLoader(DatasetBase):
    """GCC dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            max_frames=1,
            image_resolution=224,
            vit_version="ViT-B/32",
            use_felzenszwalb=False,
    ):
        super(GCC_DataLoader, self).__init__(tokenizer, max_words)
        assert max_frames == 1, "GCC dataset is an image dataset."
        self.data_path = data_path
        self.features_path = features_path
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        self.use_felzenszwalb = use_felzenszwalb and self.subset == "train"

        csv_map = {"train": "cc3m_train_desc.pkl", "val": "cc3m_val_desc.pkl", "test": None}
        assert csv_map[self.subset] is not None, "The caption file of {} is unavailable.".format(self.subset)

        data_csv = os.path.join(data_path, csv_map[self.subset])
        assert os.path.exists(data_csv), "Missed csv file, download from {}".\
            format("https://ai.google.com/research/ConceptualCaptions/download")

        features_map = {"train":"cc3m_train_lmdb_total", "val":"cc3m_val.pkl", "test":None}
        assert features_map[self.subset] is not None, "The feature of {} is unavailable.".format(self.subset)

        scale, sigma, min_size = 224, 0.9, 224
        seg_path_ = "cc3m_train_lmdb_total_seg_scale{}_sigma{}_min_size{}.lmdb".format(scale, sigma * 10, min_size)
        seg_map = {"train": seg_path_, "val": None, "test": None}
        assert seg_map[self.subset] is not None, "The feature of {} is unavailable.".format(self.subset)

        with open(data_csv, 'rb') as f:
            captions_dict_ = pickle.load(f)
        self.captions_dict = captions_dict_

        self.seg_lmdb_path = None
        self.seg_env = None
        self.seg_txn = None
        if self.use_felzenszwalb:
            seg_lmdb_path = os.path.join(features_path, seg_map[self.subset])
            self.seg_lmdb_path = seg_lmdb_path

        if self.subset == "train":
            lmdb_path = os.path.join(features_path, features_map[self.subset])
            lmdb_keys_path = os.path.join(features_path, features_map[self.subset] + "_keys.pkl")
            # env and txn is delay-loaded in ddp.
            self.lmdb_path = lmdb_path
            self.env = None
            self.txn = None
            with open(lmdb_keys_path, 'rb') as f:
                lmdb_keys = pickle.load(f)
            self.img_keys = lmdb_keys['key']
        else:
            features_path = os.path.join(self.features_path, features_map[self.subset])
            with open(features_path, 'rb') as f:
                img_data = pickle.load(f)
            self.img_data = img_data
            self.img_keys = list(self.img_data.keys())

        self.print_dist("Total Pair: {}".format(len(self.img_keys)))

        self.sample_len = len(self.img_keys)
        self.rawImageExtractor = RawImageExtractor(is_train=True, size=self.image_resolution)

    def __len__(self):
        return self.sample_len

    def _init_env(self):
        self.env = lmdb.open(self.lmdb_path, map_size=96 * 1024 * 1024 * 1024, subdir=True,
                        readonly=True, readahead=False, meminit=False, max_spare_txns=1, lock=False)
        self.txn = self.env.begin(write=False, buffers=True)

    def _init_seg_env(self):
        self.seg_env = lmdb.open(self.seg_lmdb_path, map_size=96 * 1024 * 1024 * 1024, subdir=True,
                        readonly=True, readahead=False, meminit=False, max_spare_txns=1, lock=False)
        self.seg_txn = self.seg_env.begin(write=False, buffers=True)


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.txn is not None:
            self.txn.__exit__(exc_type, exc_val, exc_tb)
        if self.env is not None:
            self.env.close()
        if self.seg_txn is not None:
            self.seg_txn.__exit__(exc_type, exc_val, exc_tb)
        if self.seg_env is not None:
            self.seg_env.close()

    def _get_rawimage(self, image_id, aug_images=False):
        # Pair x 3 x H x W, Pair is 3 as using two extra views of image
        image = np.zeros((1 if aug_images is False else 3, 3, self.image_resolution, self.image_resolution), dtype=np.float)
        coord = np.zeros((1 if aug_images is False else 3, 4), dtype=np.float)

        if self.subset == "train":
            image_bytes = self.txn.get(image_id.encode('ascii'))
        else:
            image_bytes = self.img_data[image_id]

        get_image_bool = True
        try:
            raw_image_data = self.rawImageExtractor.get_image_data_from_bytes(image_bytes, paired_aug=aug_images)
            for id_, (k_, image_data_) in enumerate(raw_image_data.items()):
                image_data_, coord_ = image_data_
                image[id_] = image_data_  # 3 x H x W
                coord[id_] = coord_  # 4
        except Exception as excep:
            self.print_dist("Raw Image reading Error in CC3M!")
            get_image_bool = False

        return image, coord, get_image_bool

    def __getitem__(self, idx):
        if self.subset == "train" and self.env is None:
            self._init_env()
        if self.subset == "train" and self.seg_lmdb_path is not None \
                and self.seg_env is None:
            self._init_seg_env()

        get_image_bool = False
        retry_num = 0
        while get_image_bool is False:
            image_id = self.img_keys[idx]
            caption = self.captions_dict[image_id]["caption"]

            pairs_text, pairs_mask, pairs_segment, choice_image_ids = self._get_text(image_id, caption)
            image, coord, get_image_bool = self._get_rawimage(image_id)

            if get_image_bool is False:
                idx = (idx + 1) % self.sample_len
            retry_num += 1
            if retry_num > 50:
                raise ValueError("Retry Limited: {}".format(retry_num))

        if self.use_felzenszwalb:
            seg4image_ = np.array(json.loads(zlib.decompress(self.seg_txn.get(image_id.encode('ascii')))), dtype=np.long)
            seg4image_ = seg4image_[2:].reshape(seg4image_[0], seg4image_[1])
            image_seg = get_felzenszwalb_from_cache(seg4image_, coord, img_size=self.image_resolution, patch_size=16)

        return_tuple = (pairs_text, pairs_mask, pairs_segment, image, coord)

        if self.use_felzenszwalb:
            return_tuple = return_tuple + (image_seg,)

        return return_tuple
