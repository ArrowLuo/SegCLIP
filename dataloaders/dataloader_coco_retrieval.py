from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import zlib
import numpy as np
import pickle
import json
import lmdb

from dataloaders.rawimage_util import RawImageExtractor
from dataloaders.rawimage_util import get_felzenszwalb_from_cache
from dataloaders.dataloader_base import DatasetBase

class COCO_DataLoader(DatasetBase):
    """COCO dataset loader."""
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
        super(COCO_DataLoader, self).__init__(tokenizer, max_words)
        assert max_frames == 1, "COCO dataset is an image dataset."
        self.data_path = data_path
        self.features_path = features_path
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.image_resolution = image_resolution

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        self.use_felzenszwalb = use_felzenszwalb and self.subset == "train"

        data_json = os.path.join(data_path, "dataset_coco.json")
        assert os.path.exists(data_json), "Missed json file, download from [karpathy split]({})"\
            .format("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip")

        with open(data_json, "r") as fp:
            captions = json.load(fp)

        # COCO dataset
        # split number: {'test': 5000, 'restval': 30504, 'val': 5000, 'train': 82783}
        # sentence number: {'test': 25010, 'restval': 152634, 'val': 25010, 'train': 414113}
        captions = captions["images"]
        subset_map = {"train":"train", "val":"val", "test":"test"}
        features_map = {"train":"coco_train2014.pkl", "val":"coco_val2014.pkl", "test":None}
        assert features_map[self.subset] is not None, "The feature of {} is unavailable.".format(self.subset)

        captions_dict = {}
        for ind, cap in enumerate(captions):
            split = cap["split"]
            if split != subset_map[self.subset]: continue
            filename = cap["filename"]
            sentences = cap["sentences"]
            captions_dict[filename] = [itm["raw"] for itm in sentences]

        features_path = os.path.join(self.features_path, features_map[self.subset])
        with open(features_path, 'rb') as f:
            img_data = pickle.load(f)
        self.img_data = img_data

        scale, sigma, min_size = 224, 0.9, 224
        seg_path_ = "coco_train2014_seg_scale{}_sigma{}_min_size{}.lmdb".format(scale, sigma * 10, min_size)
        seg_map = {"train": seg_path_, "val": None, "test": None}
        self.seg_lmdb_path = None
        self.seg_env = None
        self.seg_txn = None
        if self.use_felzenszwalb:
            seg_lmdb_path = os.path.join(self.features_path, seg_map[self.subset])
            self.seg_lmdb_path = seg_lmdb_path

        image_ids = []
        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for image_id in captions_dict.keys():
            if image_id not in self.img_data: continue
            image_ids.append(image_id)
            for cap_idx, cap_txt in enumerate(captions_dict[image_id]):
                self.sentences_dict[len(self.sentences_dict)] = (image_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.image_num: used to cut the image representation
        self.multi_sentence_per_image = True    # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.image_num = len(image_ids)
            assert len(self.cut_off_points) == self.image_num
            self.print_dist("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            self.print_dist("For {}, image number: {}".format(self.subset, self.image_num))

        self.print_dist("Image number: {}, Used number: {}".format(len(self.img_data), len(image_ids)))
        self.print_dist("Total Pair: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawImageExtractor = RawImageExtractor(is_train=True, size=self.image_resolution)

    def __len__(self):
        return self.sample_len

    def _init_seg_env(self):
        self.seg_env = lmdb.open(self.seg_lmdb_path, map_size=1 * 1024 * 1024 * 1024, subdir=True,
                        readonly=True, readahead=False, meminit=False, max_spare_txns=1, lock=False)
        self.seg_txn = self.seg_env.begin(write=False, buffers=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seg_txn is not None:
            self.seg_txn.__exit__(exc_type, exc_val, exc_tb)
        if self.seg_env is not None:
            self.seg_env.close()

    def _get_rawimage(self, image_id, aug_images=False):
        # Pair x 3 x H x W, Pair is 3 as using two extra views of image
        image = np.zeros((1 if aug_images is False else 3, 3, self.image_resolution, self.image_resolution), dtype=np.float)
        coord = np.zeros((1 if aug_images is False else 3, 4), dtype=np.float)

        raw_image_data = self.rawImageExtractor.get_image_data_from_bytes(self.img_data[image_id], paired_aug=aug_images)
        for id_, (k_, image_data_) in enumerate(raw_image_data.items()):
            image_data_, coord_ = image_data_
            image[id_] = image_data_  # 3 x H x W
            coord[id_] = coord_  # 4

        return image, coord

    def __getitem__(self, idx):
        if self.subset == "train" and self.seg_lmdb_path is not None \
                and self.seg_env is None:
            self._init_seg_env()

        image_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_image_ids = self._get_text(image_id, caption)
        image, coord = self._get_rawimage(image_id)

        if self.use_felzenszwalb:
            seg4image_ = np.array(json.loads(zlib.decompress(self.seg_txn.get(image_id.encode('ascii')))), dtype=np.long)
            seg4image_ = seg4image_[2:].reshape(seg4image_[0], seg4image_[1])
            image_seg = get_felzenszwalb_from_cache(seg4image_, coord, img_size=self.image_resolution, patch_size=16)

        return_tuple = (pairs_text, pairs_mask, pairs_segment, image, coord)

        if self.use_felzenszwalb:
            return_tuple = return_tuple + (image_seg,)

        return return_tuple
