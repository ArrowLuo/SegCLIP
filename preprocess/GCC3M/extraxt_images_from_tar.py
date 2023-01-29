import os
import tarfile
import io
from PIL import Image, TarIO
import pickle
import json

import multiprocessing
from multiprocessing import Pool

try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count


# multiprocessing.freeze_support()

def resize_image_as_bytes(image_bytes, resize_to=224):
    # restore to Image from Bytes
    image_bytes = io.BytesIO(image_bytes)
    img = Image.open(image_bytes).convert("RGB")

    w, h = img.size

    # resize image if both w and h larger than line
    if w > resize_to and h > resize_to:
        w_h_aspect_ratio = float(w) / float(h)
        assert w_h_aspect_ratio != 0.
        if w_h_aspect_ratio >= 1.:
            w, h = int(resize_to * w_h_aspect_ratio), resize_to
        else:
            w, h = resize_to, int(resize_to / w_h_aspect_ratio)

        im_resize = img.resize((w, h), Image.ANTIALIAS)
    else:
        im_resize = img

    buf = io.BytesIO()
    im_resize.save(buf, format='jpeg', quality=95)
    byte_im = buf.getvalue()
    return byte_im


def compress(paras):
    file_name, image_bytes = paras
    img_bytes = resize_image_as_bytes(image_bytes)
    return (file_name, img_bytes)


def extract_and_save(image_path_list, output_root, save_no):
    print("Total images need to process: {}".format(len(image_path_list)))
    num_works = 8
    print("Begin with {}-core logical processor.".format(num_works))

    pool = Pool(num_works)
    data_dict_list = pool.map(compress, image_path_list)
    pool.close()
    pool.join()

    imgs_dict = {}
    for img_name, img_bytes in data_dict_list:
        if img_name is None:
            print("Error: {}".format(img_bytes))
            continue
        assert img_name not in imgs_dict
        imgs_dict[img_name] = img_bytes
    print("Total images: {}".format(len(imgs_dict)))

    # For pickle
    save_path = os.path.join(output_root, "{}.pkl".format(save_no))
    pickle.dump(imgs_dict, open(save_path, "wb"))
    print("The file is saved to: {}".format(save_path))


if __name__ == "__main__":
    tar_root = "data/cc3m_train"
    output_root = "data"
    save_no = 0
    image_path_list = []
    json_dicts = {}
    for root, dirs, files in os.walk(tar_root):
        for file_no, file_name in enumerate(files):
            if ".tar" in file_name and len(file_name) == 9:
                file_path = os.path.join(root, file_name)
                print(file_path)

                filename_list = []
                with tarfile.open(file_path, "r") as tar_fp:
                    for filename in tar_fp.getmembers():
                        filename = filename.name
                        filename_list.append(filename)

                tar_fp = tarfile.open(file_path, "r")
                for filename in filename_list:
                    if ".json" in filename:
                        json_file = tar_fp.extractfile(filename)
                        """
                        {
                            "caption": "Cartoon baseball bat with the mechanic character. Vector illustration stock illustration",
                            "url": "https://thumbs.dreamstime.com/b/cartoon-baseball-bat-mechanic-character-vector-illustration-164404840.jpg",
                            "key": "012420001",
                            "status": "success",
                            "error_message": null,
                            "width": 224,
                            "height": 224,
                            "original_width": 800,
                            "original_height": 800,
                            "exif": "{}",
                            "md5": "cec60e26cb8c5a7950368823839728f4"
                        }
                        """
                        temp_dict = json.loads(json_file.read())
                        json_dicts[filename[:-5]] = temp_dict
                    elif ".jpg" in filename:
                        image_file = tar_fp.extractfile(filename)
                        image_bytes = image_file.read()
                        image_path_list.append((filename[:-4], image_bytes))
                tar_fp.close()

                if len(image_path_list) > 500000:
                    extract_and_save(image_path_list, output_root, save_no)
                    image_path_list = []
                    save_no += 1

    if len(image_path_list) > 0:
        extract_and_save(image_path_list, output_root, save_no)

    # For pickle
    save_path = os.path.join(output_root, "cc3m_train_desc.pkl")
    pickle.dump(json_dicts, open(save_path, "wb"))
    print("The file is saved to: {}".format(save_path))
