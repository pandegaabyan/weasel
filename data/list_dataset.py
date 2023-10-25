import os
from typing import Literal

import numpy as np
import torch
import random

from torch.utils import data

from sklearn.model_selection import train_test_split

from skimage import io
from skimage import measure
from skimage import transform
from skimage import morphology
from skimage import segmentation
from skimage import img_as_float
from skimage import data as skdata

'''
# ListDataset use the dataset name and task name to load the respectives images accordingly, modify it to your use case
# For a simple loader, this version is implemented considering the following folder scheme:

/datasets_root_folder
|--- dataset_1
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt

     ...
     |--- fold_k_img_list.txt
|--- dataset_2
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt

     ...
     |--- fold_k_img_list.txt
...
|--- dataset_n
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt

     ...
     |--- fold_k_img_list.txt
'''

# Constants.
root = 'openist'  # The /datasets_root_folder/ in the scheme above

annot_types = ['points', 'regions', 'skels',
               'contours', 'grid', 'dense', 'random']

all_data = [(f, f.replace('.jpg', '@.png')) for f in os.listdir('openist') if '@' not in f]

# Class that reads a sequence of image paths from a text file and creates a data.Dataset with them.


class ListDataset(data.Dataset):

    def __init__(self, mode, dataset, task, fold, resize_to, num_shots=5,
                 sparsity_mode='dense', sparsity_param=None, imgtype='med', make=True):

        assert sparsity_mode in annot_types, "{} annotation type not supported, must be one of following {}.".format(
            sparsity_mode, annot_types)

        self.imgtype = imgtype
        self.root = root
        self.num_classes = 2

        # Initializing variables.
        self.mode = mode
        self.dataset = dataset
        self.task = task
        self.fold = fold
        self.resize_to = resize_to
        self.num_shots = num_shots
        self.sparsity_mode = sparsity_mode
        self.sparsity_param = sparsity_param
        self.imgtype = imgtype

        self.sparsity_mode_list = [
            'points', 'contours', 'grid', 'regions', 'skels']
        self.imgs = None

        if make:
            # Creating list of paths.
            self.imgs = self.make_dataset()

            # Check for consistency in list.
            if len(self.imgs) == 0:

                raise (RuntimeError('Found 0 images, please check the data set'))

    # Function that create the list of pairs (img_path, mask_path)
    # Adapt this function for your dataset and fold structure
    def make_dataset(self):
        # Split the data in train/val
        tr, ts = train_test_split(all_data, test_size=0.8, random_state=self.fold, shuffle=False)

        # Select split, based on the mode
        data_list = None
        if 'train' in self.mode:
            data_list = tr
        elif 'test' in self.mode:
            data_list = ts

        random.seed(self.fold)
        random.shuffle(data_list)

        # If few-shot, select only a subset of samples
        if self.num_shots != -1 and self.num_shots <= len(data_list):
            data_list = data_list[:self.num_shots]

        items = []

        # Creating list containing image and ground truth paths.
        for it in data_list:
            item = (os.path.join(self.root, it[0]), os.path.join(self.root, it[1]))
            items.append(item)

        # Returning list.
        return items

    def make_dataset_old(self):

        # Making sure the mode is correct.
        assert self.mode in ['train', 'test', 'meta_train',
                             'meta_test', 'tune_train', 'tune_test']
        items = []

        # Setting string for the mode.
        mode_str = ''
        if 'train' in self.mode:
            mode_str = 'trn' if self.imgtype == 'med' else 'train'
        elif 'test' in self.mode:
            mode_str = 'tst' if self.imgtype == 'med' else 'val'

        # Joining input paths.
        img_path = os.path.join(self.root, self.dataset, 'images')
        msk_path = os.path.join(self.root, self.dataset,
                                'ground_truths', self.task)

        # Reading paths from file.
        data_list = [line.strip('\n') for line in open(os.path.join(
            self.root, self.dataset, self.task + '_' + mode_str + '_f' + str(self.fold) + '_few_shot.txt')).readlines()]

        random.seed(int(self.fold))
        random.shuffle(data_list)

        if self.num_shots != -1 and self.num_shots <= len(data_list):
            data_list = data_list[:self.num_shots]

        # Creating list containing image and ground truth paths.
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(msk_path, it))
            items.append(item)

        # Returning list.
        return items

    def sparse_points(self, msk, sparsity: float | Literal["random"] = 'random', index=0):
        if sparsity != 'random':
            np.random.seed(index)

        # Linearizing mask.
        msk_ravel = msk.ravel()

        # Copying raveled mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros(msk_ravel.shape[0], dtype=np.int64)
        new_msk[:] = -1

        for c in range(self.num_classes):
            # Slicing array for only containing class "c" pixels.
            msk_class = new_msk[msk_ravel == c]

            # Random permutation of class "c" pixels.
            perm = np.random.permutation(msk_class.shape[0])
            sparsity_num = round(sparsity) if sparsity != "random" else np.random.randint(low=1, high=len(perm))
            msk_class[perm[:min(sparsity_num, len(perm))]] = c

            # Merging sparse masks.
            new_msk[msk_ravel == c] = msk_class

        # Reshaping linearized sparse mask to the original 2 dimensions.
        new_msk = new_msk.reshape(msk.shape)

        return new_msk

    @staticmethod
    def sparse_grid(msk, sparsity: float | Literal["random"] = 'random', index=0):
        # Copying mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros_like(msk)
        new_msk[:, :] = -1

        if sparsity == 'random':
            # Random sparsity (x and y point spacing).
            max_high = int(np.max(msk.shape)/2)
            spacing_value = np.random.randint(low=1, high=max_high)
            spacing = (spacing_value, spacing_value)

        else:
            # Predetermined sparsity (x and y point spacing).
            spacing = (int(2 ** sparsity),
                       int(2 ** sparsity))

            np.random.seed(index)

        starting = (np.random.randint(spacing[0]),
                    np.random.randint(spacing[1]))

        new_msk[starting[0]::spacing[0], starting[1]::spacing[1]] = \
            msk[starting[0]::spacing[0], starting[1]::spacing[1]]

        return new_msk

    def sparse_contours(self, msk, sparsity: float | Literal["random"] = 'random', index=0):
        sparsity_num = sparsity if sparsity != "random" else np.random.random()

        if sparsity != 'random':
            np.random.seed(index)

        new_msk = np.zeros_like(msk)

        # Random disk radius for erosions and dilations from the original mask.
        radius_dist = np.random.randint(low=4, high=10)

        # Random disk radius for annotation thickness.
        radius_thick = 1

        # Creating morphology elements.
        selem_dist = morphology.disk(radius_dist)
        selem_thick = morphology.disk(radius_thick)

        for c in range(self.num_classes):
            # Eroding original mask and obtaining contours.
            msk_class = morphology.binary_erosion(msk == c, selem_dist)
            msk_contr = measure.find_contours(msk_class, 0.0)

            # Instantiating masks for the boundaries.
            msk_bound = np.zeros_like(msk)

            # Filling boundary masks.
            for _, contour in enumerate(msk_contr):
                rand_rot = np.random.randint(low=1, high=len(contour))
                for j, coord in enumerate(np.roll(contour, rand_rot, axis=0)):
                    if j < max(1, min(round(len(contour) * sparsity_num), len(contour))):
                        msk_bound[int(coord[0]), int(coord[1])] = c+1

            # Dilating boundary masks to make them thicker.
            msk_bound = morphology.dilation(msk_bound, footprint=selem_thick)

            # Removing invalid boundary masks.
            msk_bound = msk_bound * (msk == c)

            # Merging boundary masks.
            new_msk += msk_bound

        return new_msk - 1

    def sparse_skels(self, msk, sparsity: float | Literal["random"] = 'random', index=0):
        sparsity_num = sparsity if sparsity != "random" else np.random.random()

        bseed = None  # Blobs generator seed
        if 'tune' in self.mode:
            np.random.seed(index)
            bseed = index

        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        # Randomly selecting disk radius the annotation thickness.
        radius_thick = np.random.randint(low=1, high=2)
        selem_thick = morphology.disk(radius_thick)

        for c in range(self.num_classes):
            c_msk = (msk == c)
            c_skel = morphology.skeletonize(c_msk)
            c_msk = morphology.binary_dilation(c_skel, footprint=selem_thick)

            new_msk[c_msk] = c

        blobs = skdata.binary_blobs(np.max(new_msk.shape), blob_size_fraction=0.1,
                                    volume_fraction=sparsity_num, seed=bseed)
        blobs = blobs[:new_msk.shape[0], :new_msk.shape[1]]

        n_sp = np.zeros_like(new_msk)
        n_sp[:] = -1
        n_sp[blobs] = new_msk[blobs]

        return n_sp

    def sparse_region(self, img, msk, sparsity: float | Literal["random"] = 'random', index=0):
        # Compactness of SLIC for each dataset.
        cpn = {
            # MEDICAL
            'nih_labeled': 0.6,
            'inbreast': 0.6,
            'shenzhen': 0.7,
            'montgomery': 0.5,
            'openist': 0.5,
            'jsrt': 0.5,
            'ufba': 0.5,
            'lidc_idri_drr': 0.5,
            'panoramic': 0.75,
            'mias': 0.45
        }

        if sparsity != "random":
            # Fixing seed.
            np.random.seed(index)

        sparsity_num = sparsity if sparsity != "random" else np.random.random()

        # Copying mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        # Computing SLIC super pixels.
        slic = segmentation.slic(
            img, n_segments=250, compactness=cpn[self.dataset], start_label=1)
        labels = np.unique(slic)

        # Finding 'pure' regions, that is, the ones that only contain one label within.
        pure_regions = [[] for _ in range(self.num_classes)]
        for label in labels:
            sp = msk[slic == label].ravel()
            cnt = np.bincount(sp)

            for c in range(self.num_classes):
                if (cnt[c] if c < len(cnt) else None) == cnt.sum():
                    pure_regions[c].append(label)

        for (c, pure_region) in enumerate(pure_regions):
            # Random permutation to pure region.
            perm = np.random.permutation(len(pure_region))

            # Only keeping the selected k regions.
            perm_last_idx = max(1, round(sparsity_num * len(perm)))
            for sp in np.array(pure_region)[perm[:perm_last_idx]]:
                new_msk[slic == sp] = c

        return new_msk

    # Function to load images and masks
    # May need adaptation to your data
    # Returns: img, mask, img_filename
    def get_data(self, index):
        img_path, msk_path = self.imgs[index]

        # Reading images.
        img = io.imread(img_path)
        img = img_as_float(img)
        msk = io.imread(msk_path, as_gray=True)
        # msk = msk / np.min(np.trim_zeros(np.unique(msk)))

        img = transform.resize(img, self.resize_to, order=1, preserve_range=True)
        msk = transform.resize(msk, self.resize_to, order=0, preserve_range=True)

        img = img.astype(np.float32)
        msk[msk != 0] = 1
        msk = np.round(msk).astype(np.int64)

        # Splitting path.
        img_filename = os.path.split(img_path)[-1]

        # remove extension from filename
        img_filename = ".".join(img_filename.split(".")[:-1])

        return img, msk, img_filename

    @staticmethod
    def norm(img):
        normalized = np.zeros(img.shape)
        if len(img.shape) == 2:
            normalized = (img - img.mean()) / img.std()
        else:
            for b in range(img.shape[2]):
                normalized[:, :, b] = (img[:, :, b] - img[:, :, b].mean()) / img[:, :, b].std()
        return normalized

    @staticmethod
    def torch_channels(img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, -1, 0)
        return img

    def __getitem__(self, index):

        img, msk, img_filename = self.get_data(index)

        sparse_msk = np.copy(msk)

        if self.sparsity_mode == 'random':

            # Randomly selecting sparsity mode.
            sparsity = np.random.randint(0, len(self.sparsity_mode_list))

            if sparsity == 0:
                sparse_msk = self.sparse_points(
                    msk, sparsity='random', index=index)
            elif sparsity == 1:
                sparse_msk = self.sparse_contours(
                    msk, sparsity='random', index=index)
            elif sparsity == 2:
                sparse_msk = self.sparse_grid(
                    msk, sparsity='random', index=index)
            elif sparsity == 3:
                sparse_msk = self.sparse_region(
                    img, msk, sparsity='random', index=index)
            elif sparsity == 4:
                sparse_msk = self.sparse_skels(
                    msk, sparsity='random', index=index)

        # Randomly selecting sparse points.
        elif self.sparsity_mode == 'points':
            sparse_msk = self.sparse_points(
                msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'contours':
            sparse_msk = self.sparse_contours(
                msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'grid':
            sparse_msk = self.sparse_grid(
                msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'regions':
            sparse_msk = self.sparse_region(
                img, msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'skels':
            sparse_msk = self.sparse_skels(
                msk, sparsity=self.sparsity_param, index=index)

        # Normalization.
        img = self.norm(img)

        # Adding channel dimension.
        img = self.torch_channels(img)

        # Turning to tensors.
        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk).type(torch.LongTensor)

        sparse_msk = torch.from_numpy(sparse_msk).type(torch.LongTensor)

        # Returning to iterator.
        return img, msk, sparse_msk, img_filename

    def __len__(self):
        return len(self.imgs)
