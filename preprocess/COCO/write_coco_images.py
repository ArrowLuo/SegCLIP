import os
import io
import pickle
import argparse
from PIL import Image

import multiprocessing
from multiprocessing import Pool

try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count


# multiprocessing.freeze_support()

def resize_image_as_bytes(image_path, resize_to=224):
    img = Image.open(image_path)
    w, h = img.size

    # resize image if both w and h larger than exception
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
    im_resize.save(buf, format=img.format)
    byte_im = buf.getvalue()
    return byte_im

def compress(paras):
    file_name, image_path = paras
    try:
        img_bytes = resize_image_as_bytes(image_path)
    except Exception as e:
        return (None, file_name + ":" + str(e))
    return (file_name, img_bytes)


def make_pickle(input_root, output_root, split="train2014"):
    """
    :param input_root:
    :param output_root:
    :param split: train2014, val2014
    :return:
    """
    img_dir = os.path.join(input_root, split)
    assert os.path.exists(img_dir)
    os.makedirs(output_root, exist_ok=True)

    image_path_list = []
    for root, dirs, files in os.walk(img_dir):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            image_path_list.append((file_name, image_path))

    print("Total images need to process: {}".format(len(image_path_list)))
    num_works = cpu_count()
    print("Begin with {}-core logical processor.".format(num_works))

    pool = Pool(num_works)
    data_dict_list = pool.map(compress, image_path_list)
    pool.close()
    pool.join()

    imgs_dict = {}
    for file_name, img_bytes in data_dict_list:
        if file_name is None:
            print("Error: {}".format(img_bytes))
            continue
        assert file_name not in imgs_dict
        imgs_dict[file_name] = img_bytes
    print("Total images: {}".format(len(imgs_dict)))

    # For pickle
    save_path = os.path.join(output_root, "{}.pkl".format(split))
    pickle.dump(imgs_dict, open(save_path, "wb"))
    print("The file is saved to: {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress images for speed-up')
    parser.add_argument('--input_root', type=str, help='input root')
    parser.add_argument('--output_root', type=str, help='output root')
    parser.add_argument('--data_split', type=str, help='data split, e.g., train2014, val2014')
    args = parser.parse_args()
    make_pickle(args.input_root, args.output_root, split=args.data_split)
