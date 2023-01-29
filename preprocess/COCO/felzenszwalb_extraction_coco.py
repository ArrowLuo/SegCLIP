import io
import os
import time
import json
import pickle
from PIL import Image, ImageFilter
from skimage import segmentation
import numpy as np
import zlib

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Queue, Process
import lmdb

try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
# multiprocessing.freeze_support()

# Handle the GPU issue in multi-processing.
from multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def processer(img_data, map_size, input_queue, output_queue, scale, sigma, min_size):
    while True:
        image_id, finished = input_queue.get()
        if not finished:
            image_bytes = io.BytesIO(img_data[image_id])
            restore_img = Image.open(image_bytes).convert("RGB")
            # restore_img: (width, height, 3)
            seg_map_ = segmentation.felzenszwalb(restore_img, scale=scale, sigma=sigma, min_size=min_size)

            h_, w_ = seg_map_.shape
            seg_map_ = seg_map_.flatten()
            seg_map_ = np.concatenate(([h_, w_], seg_map_), axis=0)
            seg_map_ = json.dumps(seg_map_.tolist()).encode()
            seg_map_ = zlib.compress(seg_map_)

            # Restore with
            # image_np = np.array(json.loads(zlib.decompress(value)), dtype=np.long)
            # image_np = image_np[2:].reshape(image_np[0], image_np[1])
            # print(image_np)

            output_queue.put((image_id, seg_map_))
        else:
            break


def reducer(lmdb_save_path, save_map_size, output_queue):
    env_save = lmdb.open(lmdb_save_path, map_size=save_map_size, subdir=True)
    txn_save = env_save.begin(write=True)
    while True:
        image_id, seg_map = output_queue.get()
        if seg_map is not None:
            txn_save.put(key=image_id.encode('ascii'), value=seg_map)
        else:
            txn_save.commit()
            env_save.close()
            break


def setup_mp(image_ids, img_data, lmdb_save_path, map_size, save_map_size, scale, sigma, min_size, num_works=16):
    QUEUE_SIZE = 10000
    input_queue = Queue(maxsize=QUEUE_SIZE)
    output_queue = Queue(maxsize=QUEUE_SIZE)

    workers = []
    for worker_id in range(num_works):
        worker = Process(target=processer, args=(img_data, map_size, input_queue, output_queue, scale, sigma, min_size))
        worker.daemon = True
        worker.start()
        workers.append(worker)

    reduce = Process(target=reducer, args=(lmdb_save_path, save_map_size, output_queue,))
    reduce.start()

    for idx, image_id in enumerate(image_ids):
        input_queue.put((image_id, False))
        if idx % 1000 == 0:
            print("{}/{}".format(idx, len(image_ids)))

    # Notifying workers the end of input
    for _ in workers:
        input_queue.put((-1, True))

    # wait for workers to terminate
    for w in workers:
        w.join()

    # Notify the reducer the end of output
    output_queue.put((-1, None))

    # wait for reducer to terminate
    reduce.join()


if __name__ == "__main__":
    features_path = "data/COCO"

    scale, sigma, min_size = 224, 0.9, 224
    features_map = {"train": "coco_train2014.pkl", "val": "coco_val2014.pkl", "test": None}
    map_size = 0
    save_map_size = 1 * 1024 * 1024 * 1024

    for subset in ["train"]:
        assert features_map[subset] is not None, "The feature of {} is unavailable.".format(subset)

        pickle_path = os.path.join(features_path, features_map[subset])

        with open(pickle_path, 'rb') as f:
            img_data = pickle.load(f)
        img_keys = img_data.keys()

        lmdb_save_path = "_seg_scale{}_sigma{}_min_size{}.lmdb".format(scale, sigma * 10, min_size)
        lmdb_save_path = pickle_path.replace(".pkl", "") + lmdb_save_path

        print("Total images need to process: {}".format(len(img_keys)))
        num_works = 32
        print("Begin with {}-core logical processor.".format(num_works))
        setup_mp(img_keys, img_data, lmdb_save_path, map_size, save_map_size, scale, sigma, min_size, num_works=num_works)

        print("lmdb has been saved in {}".format(lmdb_save_path))
