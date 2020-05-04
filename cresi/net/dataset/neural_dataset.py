import os
import sys
import random
import numpy as np

# import relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from augmentations.composition import Compose
from augmentations.transforms import ToTensor
from dataset.abstract_image_provider import AbstractImageProvider
from .image_cropper import ImageCropper


class Dataset:
    """
    base class for datasets. for every image from image provider you will 
    have its own cropper
    """
    def __init__(self, image_provider: AbstractImageProvider, image_indexes, 
                 config, stage='train', transforms=None, verbose=False):
        self.pad = 0 if stage=='train' else config.test_pad
        self.image_provider = image_provider
        self.image_indexes = image_indexes if isinstance(image_indexes, list) \
                        else image_indexes.tolist()
        if verbose:
            print("nueral_dataset.py - Dataset - len image_indexes:", len(self.image_indexes))
        self.stage = stage
        self.keys = {'image', 'image_name'}
        self.config = config
        self.transforms = Compose([transforms, ToTensor(config.num_classes)])
        self.croppers = {}

    def __getitem__(self, item):
        raise NotImplementedError

    def get_cropper(self, image_id, val=False):
        #todo maybe cache croppers for different sizes too speedup if it's slow part?
        if image_id not in self.croppers:
            image = self.image_provider[image_id].image
            rows, cols = image.shape[:2]
            if self.config.ignore_target_size and val:
                # we can igore target size if we want to validate on full images
                assert self.config.predict_batch_size == 1
                target_rows, target_cols = rows, cols
            else:
                target_rows, target_cols = self.config.target_rows, self.config.target_cols
            cropper = ImageCropper(rows, cols,
                                   target_rows, target_cols,
                                   self.pad)
            self.croppers[image_id] = cropper
        return self.croppers[image_id]


class TrainDataset(Dataset):
    """
    dataset for training with random crops
    """
    def __init__(self, image_provider, image_indexes, config, stage='train', 
                 transforms=None, partly_sequential=False, verbose=False):
        super(TrainDataset, self).__init__(image_provider, image_indexes, 
             config, stage, transforms=transforms)
        self.keys.add('mask')
        self.partly_sequential = partly_sequential
        self.inner_idx = 9
        self.idx = 0
        if verbose:
            print("nueral_dataset.py - TrainDataset - len imaage_indexes:", len(image_indexes))

    def __getitem__(self, idx, verbose=False):
        if self.partly_sequential:
            if verbose:
                print("nueral_dataset.py - TrainDataset - __getitem__ partly_sequential:", self.partly_sequential)
            # use this if your images are too big
            if self.inner_idx > 8:
                self.idx = idx
                self.inner_idx = 0
            self.inner_idx += 1
            im_idx = self.image_indexes[self.idx % len(self.image_indexes)]
        else:
            im_idx = self.image_indexes[idx % len(self.image_indexes)]

        if verbose:
            print("nueral_dataset.py - TrainDataset - __getitem__ im_idx:", im_idx)

        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        sx, sy = cropper.random_crop_coords()
        if cropper.use_crop and self.image_provider.has_alpha:
            for i in range(10):
                alpha = cropper.crop_image(item.alpha, sx, sy)
                if np.mean(alpha) > 5:
                    break
                sx, sy = cropper.random_crop_coords()
            else:
                return self.__getitem__(random.randint(0, len(self.image_indexes)))

        im = cropper.crop_image(item.image, sx, sy)
        if not np.any(im > 5):
            # re-try random if image is empty
            return self.__getitem__(random.randint(0, len(self.image_indexes)))
        mask = cropper.crop_image(item.mask, sx, sy)
        data = {'image': im, 'mask': mask, 'image_name': item.fn}

        #print ("neural_dataset.py data:", data)

        return self.transforms(**data)

    def __len__(self, verbose=False):
        z = len(self.image_indexes) * max(self.config.epoch_size, 1)
        if verbose:
            print("nueral_dataset.py - TrainDataset - __len__:", z)
        return z # epoch size is len images

class SequentialDataset(Dataset):
    """
    dataset for test and base dataset for validation.
    produces sequential crops of images
    """
    def __init__(self, image_provider, image_indexes, config, stage='test', transforms=None):
        super(SequentialDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.good_tiles = []
        self.init_good_tiles()
        self.keys.update({'geometry'})

    def init_good_tiles(self, verbose=False):
        self.good_tiles = []
        for im_idx in self.image_indexes:
            cropper = self.get_cropper(im_idx, val=True)
            positions = cropper.positions
            if self.image_provider.has_alpha:
                item = self.image_provider[im_idx]
                alpha_generator = cropper.sequential_crops(item.alpha)
                for idx, alpha in enumerate(alpha_generator):
                    if np.mean(alpha) > 5:
                        self.good_tiles.append((im_idx, *positions[idx]))
            else:
                for pos in positions:
                    self.good_tiles.append((im_idx, *pos))
        if verbose:
            print("neural_dataset.py - SequentialDataset -  prepare_image() - init_good_tiles() - good_tiles", self.good_tiles)

    def prepare_image(self, item, cropper, sx, sy, verbose=False):
        im = cropper.crop_image(item.image, sx, sy)
        rows, cols = item.image.shape[:2]
        geometry = {'rows': rows, 'cols': cols, 'sx': sx, 'sy': sy}
        data = {'image': im, 'image_name': item.fn, 'geometry': geometry}
        if verbose:
            print("neural_dataset.py - SequentialDataset -  prepare_image() - image_name:", item.fn)
            print("neural_dataset.py - SequentialDataset -  prepare_image() - geometry:", geometry)
            print("neural_dataset.py - SequentialDataset -  prepare_image() - im.shape:", im.shape)
        return data

    def __getitem__(self, idx, verbose=False):
        if idx >= self.__len__():
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        data = self.prepare_image(item, cropper, sx, sy)
        if verbose:
            print("neural_dataset.py - SequentialDataset -  __get_item__() - data['image'].shape:", data['image'].shape)

        return self.transforms(**data)

    def __len__(self):
        return len(self.good_tiles)


class ValDataset(SequentialDataset):
    """
    same as sequential but added mask
    """
    def __init__(self, image_provider, image_indexes, config, stage='train', transforms=None):
        super(ValDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')

    def __getitem__(self, idx, verbose=False):
        im_idx, sx, sy = self.good_tiles[idx]
        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        data = self.prepare_image(item, cropper, sx, sy)
        if verbose:
            print("neural_dataset.py - ValDataset -  __get_item__() - data['image'].shape:", data['image'].shape)


        mask = cropper.crop_image(item.mask, sx, sy)
        data.update({'mask': mask})
        return self.transforms(**data)
