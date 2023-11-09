import munch
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import random, copy

_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_SPLITS = ('train', 'val', 'test')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = os.path.join(metadata_root,
                                            'image_ids_proxy.txt')
    metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
    metadata.localization = os.path.join(metadata_root, 'localization.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels


def get_bounding_boxes(metadata):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
            x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, x1, y0, y1))
            else:
                boxes[image_id] = [(x0, x1, y0, y1)]
    return boxes


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


class WSOLImageLabelDataset(Dataset):
    def __init__(self, data_root, metadata_root, transform, proxy,
                 num_sample_per_class=0):
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class

        self._adjust_samples_per_class()

    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            if len(indices) < self.num_sample_per_class:
                raise ValueError(
                    "The number of indices is smaller than num_sample_per_class (# of indices: {}, num_sample_per_class: {})".format(
                        len(indices), self.num_sample_per_class))
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')
        image = self.transform(image)
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


def get_data_loader(data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, proxy_training_set,
                    num_val_sample_per_class=0, load_cross_image=False):
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]))

    loader_dict = {
        'train': WSOLImageLabelDataset,
        'val': WSOLImageLabelDataset,
        'test': WSOLImageLabelDataset
    }

    loaders = {
        split: DataLoader(
            loader_dict[split](
                data_root=data_roots[split],
                metadata_root=os.path.join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == 'train',
                num_sample_per_class=(num_val_sample_per_class
                                      if split == 'val' else 0)
            ),
            batch_size=batch_size,
            shuffle=(split == 'train') and not load_cross_image,
            num_workers=workers)
        for split in _SPLITS
    }
    return loaders