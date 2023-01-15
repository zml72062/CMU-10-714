import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        # BEGIN YOUR SOLUTION
        return img[Ellipsis, :, ::-1, :] if flip_img else img
        # END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding+1, size=2)
        # BEGIN YOUR SOLUTION
        out_img = img
        shape = list(img.shape)
        if self.padding > 0:
            shapelr = shape[:]
            shapelr[-2] = self.padding
            lr = np.zeros(shapelr)

            shapeud = shape[:]
            shapeud[-3] = self.padding
            shapeud[-2] += 2*self.padding
            ud = np.zeros(shapeud)

            out_img = np.concatenate((
                ud, np.concatenate((
                    lr, img, lr
                ), axis=1), ud
            ), axis=0)
        shift_x += self.padding
        shift_y += self.padding
        return out_img[Ellipsis, shift_x:shift_x+img.shape[-3], shift_y:shift_y+img.shape[-2], :]

        # END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.ordering = np.array_split(np.arange(len(dataset)),
                                       range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        # BEGIN YOUR SOLUTION
        order = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(order)
            self.ordering = [order[s] for s in self.ordering]
        self.__dict__.setdefault('batch_idx', 0)
        return self
        # END YOUR SOLUTION

    def __next__(self):
        # BEGIN YOUR SOLUTION
        batch_idx = self.__dict__['batch_idx']
        self.__dict__['batch_idx'] = batch_idx + 1
        try:
            import needle as ndl
            return tuple(ndl.Tensor(elem) for elem in self.dataset[self.ordering[batch_idx]])
        except IndexError:
            raise StopIteration
        # END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # BEGIN YOUR SOLUTION
        import gzip
        import struct
        with gzip.open(image_filename) as f:
            content = f.read()
            magic, length, row, column = struct.unpack(">iiii", content[:16])
            X = struct.unpack(">"+"B"*length*row*column, content[16:])
            X = np.array(X, dtype=np.float32).reshape(length, row*column)
            X = X/X.max()
            X = X.reshape(length, row, column, 1)

        with gzip.open(label_filename) as f:
            content = f.read()
            magic, length = struct.unpack(">ii", content[:8])
            y = struct.unpack(">"+"B"*length, content[8:])
            y = np.array(y, dtype=np.uint8)

        self.length = length
        self.X = X
        self.y = y
        self.transforms = transforms
        self.num_features = row * column
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # BEGIN YOUR SOLUTION
        X_select = self.X[index]
        y_select = self.y[index]
        if self.transforms is not None:
            for trans in self.transforms:
                X_select = trans(X_select)

        return X_select, y_select
        # END YOUR SOLUTION

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return self.length
        # END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
