from __future__ import division

import torch
import math
import random
import numpy as np
# pip install Pillow==8.2.0
from PIL import Image, ImageFilter
try:
    import accimage
except ImportError:
    accimage = None
import warnings

from skimage import segmentation
from einops import rearrange

from torchvision.transforms import functional as F
## Rewrite Compose, RandomResizedCrop, and RandomHorizontalFlip to return crop coordinates:
## rates of (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize # Compose, RandomResizedCrop
from torchvision.transforms import ColorJitter, RandomApply, RandomGrayscale # RandomHorizontalFlip
import io

SLIP_DEFAULT_MEAN = [0.485, 0.456, 0.406]
SLIP_DEFAULT_STD = [0.229, 0.224, 0.225]
CLIP_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]

CHOICE_MEAN = CLIP_DEFAULT_MEAN
CHOICE_STD = CLIP_DEFAULT_STD

class RawImageExtractor():
    def __init__(self, is_train=False, size=224, ):
        self.is_train = is_train
        self.size = size
        self.transform = self._transform(self.size, self.is_train)
        self.transform_augment = self._transform_augment(self.size)

    def _transform(self, n_px, is_train=False):
        normalize = Normalize(mean=CHOICE_MEAN, std=CHOICE_STD)   # In CLIP

        transform_list_ = []
        if is_train:
            transform_list_ += [RandomResizedCropCoord(n_px, scale=(0.5, 1.0), interpolation=Image.BICUBIC)]
        else:
            transform_list_ += [Resize(n_px, interpolation=Image.BICUBIC), CenterCrop(n_px)]

        transform_list_ += [lambda image: image.convert("RGB"), ToTensor(), normalize]
        _transform = ComposeCoord(transform_list_)

        return _transform


    def _transform_augment(self, n_px):
        normalize = Normalize(mean=CHOICE_MEAN, std=CHOICE_STD)
        augment = ComposeCoord([
            RandomResizedCropCoord(n_px, scale=(0.5, 1.)),
            RandomApply([
                ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            lambda image: image.convert("RGB"),
            ToTensor(),
            normalize,
        ])
        return augment

    def get_image_data_from_bytes(self, image_bytes, paired_aug=False):
        """
        :param image_bytes:
            The image_bytes is genetate as follows,
                buf = io.BytesIO()
                im_resize.save(buf, format=img.format)
                byte_im = buf.getvalue()
        :return:
        """

        # restore to Image from Bytes
        image_bytes = io.BytesIO(image_bytes)
        # image_bytes.seek(0)
        restore_img = Image.open(image_bytes).convert("RGB")
        image_data = self.transform(restore_img)

        ret_data = {'image': image_data}
        if paired_aug:
            image_aug1 = self.transform_augment(restore_img)
            image_aug2 = self.transform_augment(restore_img)
            ret_data['image_aug1'] = image_aug1
            ret_data['image_aug2'] = image_aug2

        return ret_data

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

def get_felzenszwalb_from_cache(image_seg_pic, coord, img_size=224, patch_size=16):
    h_pic, w_pic = image_seg_pic.shape

    seg_map_tensor_ls = []
    # coord: (T, 4)

    for coo_ in coord:
        # (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
        x_upper_left, y_upper_left, x_lower_right, y_lower_right = coo_

        flip_h, flip_v = False, False
        if x_upper_left > x_lower_right:
            x_upper_left, x_lower_right = x_lower_right, x_upper_left
            flip_h = True
        if y_upper_left > y_lower_right:
            y_upper_left, y_lower_right = y_lower_right, y_upper_left
            flip_v = True
        
        # NOTE: use math.ceil to avoid empty element
        x_upper_left, x_lower_right = int(x_upper_left * w_pic), math.ceil(x_lower_right * w_pic)
        y_upper_left, y_lower_right = int(y_upper_left * h_pic), math.ceil(y_lower_right * h_pic)

        if y_lower_right - y_upper_left < 2 or x_lower_right - x_upper_left < 2:
            seg_map_ = image_seg_pic
        else:
            seg_map_ = image_seg_pic[y_upper_left:y_lower_right, x_upper_left:x_lower_right]

        if flip_h: seg_map_ = np.flip(seg_map_, axis=1)
        if flip_v: seg_map_ = np.flip(seg_map_, axis=0)

        seg_map_ = torch.tensor(seg_map_)
        seg_map_ = seg_map_.reshape(1, 1, seg_map_.size(0), seg_map_.size(1))
        seg_map_ = torch.nn.functional.interpolate(seg_map_.to(torch.float), size=img_size, mode='nearest')
        seg_map_ = seg_map_.squeeze().to(torch.long).cpu().numpy()

        patch_len = img_size // patch_size
        seg_map_ = rearrange(seg_map_, '(x1 p1) (x2 p2)-> (x1 x2) (p1 p2)', x1=patch_len, x2=patch_len, p1=patch_size, p2=patch_size)
        seg_map_ = np.mean(seg_map_, axis=-1)
        seg_map_ = rearrange(seg_map_, '(x1 x2) -> x1 x2', x1=patch_len, x2=patch_len)
        seg_map_ = seg_map_.astype(dtype=np.long)
        seg_map_tensor_ls.append(seg_map_)

    image_seg = np.stack(seg_map_tensor_ls, axis=0)

    return image_seg

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

"""
Below classes are adapted from: https://github.com/zdaxie/PixPro
"""
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class ComposeCoord(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        coord = None
        for t in self.transforms:
            if 'RandomResizedCropCoord' in t.__class__.__name__:
                img, coord = t(img)
            elif 'FlipCoord' in t.__class__.__name__:
                img, coord = t(img, coord)
            else:
                img = t(img)

        # For uniform return in batch manner.
        if coord is None:
            coord = torch.Tensor([0., 0., 0., 0.])

        return img, coord

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomHorizontalFlipCoord(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            return F.hflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipCoord(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[1] = coord[3]
            coord_new[3] = coord[1]
            return F.vflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCropCoord(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)
        if width - 1 == 0 or height - 1 == 0:
            coord = torch.Tensor([0., 0., 0., 0.])
            # warnings.warn("RandomResizedCropCoord width is {}, height is {}".format(width, height))
        else:
            coord = torch.Tensor([float(j) / (width - 1), float(i) / (height - 1),
                                  float(j + w - 1) / (width - 1), float(i + h - 1) / (height - 1)])
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), coord

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string



