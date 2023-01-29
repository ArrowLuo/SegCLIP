import os
import numpy as np
import pickle
import json
from collections import defaultdict
import lmdb
import io
from PIL import Image
import shutil

if __name__ == "__main__":
    lmdb_path = "data"
    lmdb_file = "cc3m_train_lmdb_total"

    bool_write = True
    if bool_write:
        lmdb_save_path = os.path.join(lmdb_path, lmdb_file)
        keys_save_path = os.path.join(lmdb_path, lmdb_file + "_keys.pkl")

        assert os.path.exists(lmdb_save_path) is False, "Exsiting~"
        assert os.path.exists(keys_save_path) is False, "Exsiting~"

        # clean previous version
        if os.path.exists(lmdb_save_path):
            shutil.rmtree(lmdb_save_path)
        if os.path.exists(keys_save_path):
            os.remove(keys_save_path)

        image_keys = defaultdict(list)
        image_names_ = []

        env = lmdb.open(lmdb_save_path, map_size=96 * 1024 * 1024 * 1024, subdir=True)
        txn = env.begin(write=True)

        features_path = os.path.join(lmdb_path, "cc3m_train.pkl")
        with open(features_path, 'rb') as f:
            img_data = pickle.load(f)

        image_names_ = image_names_ + list(img_data.keys())
        print("Images len: {}".format(len(image_names_)))

        for image_name_, image_bytes in img_data.items():
            txn.put(key=image_name_.encode('ascii'), value=image_bytes)

        txn.commit()
        env.close()
        print("lmdb has been saved in {}".format(lmdb_save_path))

        image_keys['key'] = image_names_
        pickle.dump(image_keys, open(keys_save_path, "wb"))
        print("The file is saved to: {}".format(keys_save_path))

    else:
        # ===============================
        # Read for test
        # ===============================
        assert os.path.exists(os.path.join(lmdb_path, lmdb_file)) is True
        env = lmdb.open(os.path.join(lmdb_path, lmdb_file), map_size=350 * 1024 * 1024 * 1024, subdir=True,
                        readonly=True, readahead=False, meminit=False, max_spare_txns=1, lock=False)
        txn = env.begin(write=False, buffers=True)

        image_bytes = txn.get("00414d3a7b400d280d9cbe92f6ca2".encode('ascii'))
        # restore_img to Image from Bytes
        image_bytes = io.BytesIO(image_bytes)
        restore_img = Image.open(image_bytes).convert("RGB")
        print(restore_img.size)

        for key, value in txn.cursor():
            print(str(key, 'ascii'))
            image_bytes = value.tobytes()
            # restore to Image from Bytes
            image_bytes = io.BytesIO(image_bytes)
            restore_img = Image.open(image_bytes).convert("RGB")
            print(restore_img.size)
            break
        env.close()