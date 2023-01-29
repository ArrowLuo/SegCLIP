import io
import os
import time
import json
import pickle
from PIL import Image, ImageFilter
from skimage import segmentation
from einops import rearrange
import cv2
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


def processer(lmdb_path, map_size, input_queue, output_queue, scale, sigma, min_size):
    # env and txn is delay-loaded in ddp. They can't pickle
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=True,
                    readonly=True, readahead=False, meminit=False, max_spare_txns=1, lock=False)
    txn = env.begin(write=False, buffers=True)
    while True:
        image_id, finished = input_queue.get()
        if not finished:
            image_bytes = io.BytesIO(txn.get(image_id.encode('ascii')))
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
            env.close()
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

def setup_mp(image_ids, lmdb_path, lmdb_save_path, map_size, save_map_size, scale, sigma, min_size, num_works=16):
    QUEUE_SIZE = 10000
    input_queue = Queue(maxsize=QUEUE_SIZE)
    output_queue = Queue(maxsize=QUEUE_SIZE)

    workers = []
    for worker_id in range(num_works):
        worker = Process(target=processer, args=(lmdb_path, map_size, input_queue, output_queue, scale, sigma, min_size))
        worker.daemon = True
        worker.start()
        workers.append(worker)

    reduce = Process(target=reducer, args=(lmdb_save_path, save_map_size, output_queue,))
    reduce.start()

    for idx, image_id in enumerate(image_ids):
        input_queue.put((image_id, False))
        if idx % 10000 == 0:
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
    features_path = "data"

    scale, sigma, min_size = 224, 0.9, 224
    features_map = {"train":"cc3m_train_lmdb_total", "val":"cc3m_val.pkl", "test":None}
    map_size = 96 * 1024 * 1024 * 1024
    save_map_size = 96 * 1024 * 1024 * 1024

    for subset in ["train"]:
        assert features_map[subset] is not None, "The feature of {} is unavailable.".format(subset)

        lmdb_path = os.path.join(features_path, features_map[subset])
        lmdb_keys_path = os.path.join(features_path, features_map[subset] + "_keys.pkl")

        with open(lmdb_keys_path, 'rb') as f:
            lmdb_keys = pickle.load(f)
        img_keys = lmdb_keys['key']

        lmdb_save_path = lmdb_path + "_seg_scale{}_sigma{}_min_size{}.lmdb".format(scale, sigma * 10, min_size)

        print("Total images need to process: {}".format(len(img_keys)))
        num_works = 64
        print("Begin with {}-core logical processor.".format(num_works))
        setup_mp(img_keys, lmdb_path, lmdb_save_path, map_size, save_map_size, scale, sigma, min_size, num_works=num_works)

        print("lmdb has been saved in {}".format(lmdb_save_path))

