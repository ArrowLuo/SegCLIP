import torch
from torch.utils.data import ConcatDataset, DataLoader
from dataloaders.dataloader_coco_retrieval import COCO_DataLoader
from dataloaders.dataloader_cc_retrieval import GCC_DataLoader
from dataloaders.data_config import DATA_CONFIG_DICT
# pip install prefetch_generator
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        # transforms generator into a background-thead generator.
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)

DATALOADER_FCT_DICT_ = {}
DATALOADER_FCT_DICT_["coco"] = COCO_DataLoader
DATALOADER_FCT_DICT_["cc"] = GCC_DataLoader

def _get_dataset(args, tokenizer, dataloader_fct, data_path, features_path, subset="train"):
    dataset = dataloader_fct(
        subset=subset,
        data_path=data_path,
        features_path=features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=1,
        image_resolution=224,
        vit_version=args.pretrained_clip_name,
        use_felzenszwalb=args.use_seglabel,
    )
    return dataset

def _train_sampler_dataloader(args, dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoaderX(
        dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True if args.use_pin_memory else False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return train_dataloader, train_sampler

def _test_sampler_dataloader(args, dataset):
    test_dataloader = DataLoaderX(
        dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=True if args.use_pin_memory else False,
        shuffle=False,
        drop_last=False,
    )
    return test_dataloader

def _get_train_dataloader_fct(data_name):
    def _train_dataloader_fct(args, tokenizer):
        assert data_name in DATALOADER_FCT_DICT_, "{} not in DATALOADER_FCT_DICT".format(data_name)
        assert data_name in DATA_CONFIG_DICT, "{} not in DATA_CONFIG_DICT".format(data_name)
        dataloader_fct = DATALOADER_FCT_DICT_[data_name]
        data_path = DATA_CONFIG_DICT[data_name]["train"]["data_path"]
        features_path = DATA_CONFIG_DICT[data_name]["train"]["features_path"]
        dataset = _get_dataset(args, tokenizer, dataloader_fct, data_path, features_path, subset="train")
        train_dataloader, train_sampler = _train_sampler_dataloader(args, dataset)
        return train_dataloader, len(dataset), train_sampler
    return _train_dataloader_fct

def _get_test_dataloader_fct(data_name):
    def _test_dataloader_fct(args, tokenizer, subset="test"):
        assert data_name in DATALOADER_FCT_DICT_, "{} not in DATALOADER_FCT_DICT".format(data_name)
        assert data_name in DATA_CONFIG_DICT, "{} not in DATA_CONFIG_DICT".format(data_name)
        dataloader_fct = DATALOADER_FCT_DICT_[data_name]
        data_path = DATA_CONFIG_DICT[data_name][subset]["data_path"]
        features_path = DATA_CONFIG_DICT[data_name][subset]["features_path"]
        testset = _get_dataset(args, tokenizer, dataloader_fct, data_path, features_path, subset=subset)
        test_dataloader = _test_sampler_dataloader(args, testset)
        return test_dataloader, len(testset)
    return _test_dataloader_fct

def _get_train_multi_dataloader_fct(data_name):
    def _train_dataloader_fct(args, tokenizer):
        data_name_list = data_name.split(",")
        dataset_list = []
        for data_name_ in data_name_list:
            if len(data_name_) == 0: continue
            assert data_name_ in DATALOADER_FCT_DICT_, "{} not in DATALOADER_FCT_DICT".format(data_name_)
            assert data_name_ in DATA_CONFIG_DICT, "{} not in DATA_CONFIG_DICT".format(data_name_)
            dataloader_fct = DATALOADER_FCT_DICT_[data_name_]
            data_path = DATA_CONFIG_DICT[data_name_]["train"]["data_path"]
            features_path = DATA_CONFIG_DICT[data_name_]["train"]["features_path"]
            dataset_ = _get_dataset(args, tokenizer, dataloader_fct, data_path, features_path, subset="train")
            dataset_list.append(dataset_)

        dataset = ConcatDataset(dataset_list)
        train_dataloader, train_sampler = _train_sampler_dataloader(args, dataset)
        return train_dataloader, len(dataset), train_sampler
    return _train_dataloader_fct

dataloader_cc_train = _get_train_dataloader_fct("cc")
dataloader_cc_test = _get_test_dataloader_fct("cc")

dataloader_coco_train = _get_train_dataloader_fct("coco")
dataloader_coco_test = _get_test_dataloader_fct("coco")

class DataloaderDictClass(dict):
    def __getitem__(self, item):
        if item not in self and item.find(",") > -1:
            train_loader_ = _get_train_multi_dataloader_fct(item)
            v = {"train": train_loader_, "val": None, "test": None}
        else:
            v = super(DataloaderDictClass, self).__getitem__(item)
        return v

DATALOADER_DICT = DataloaderDictClass()
DATALOADER_DICT["cc"] = {"train":dataloader_cc_train, "val":dataloader_cc_test, "test":None}
DATALOADER_DICT["coco"] = {"train":dataloader_coco_train, "val":dataloader_coco_test, "test":None}
