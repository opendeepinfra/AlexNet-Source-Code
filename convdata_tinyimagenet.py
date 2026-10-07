"""Tiny-ImageNet-200 data provider for the 2012 AlexNet port.

Tiny-ImageNet-200 is 200 classes of 64x64 RGB images (100k train / 10k val),
which is the largest ImageNet-derived set that can be trained end-to-end on a
single GPU. It is served from the flat ``.npy`` caches produced by
``make-data/prepare_tinyimagenet.py``:

    tiny-imagenet-train.npy          uint8   (100000, 3, 64, 64)
    tiny-imagenet-train-labels.npy   int16   (100000,)
    tiny-imagenet-val.npy            uint8   (10000, 3, 64, 64)
    tiny-imagenet-val-labels.npy     int16   (10000,)
    tiny-imagenet-mean.npy           float32 (3*64*64, 1)
    tiny-imagenet-classes.txt        200 lines "<label_index> <wnid>"

The caches are memory-mapped, so the provider costs almost no RAM and each
epoch is a memcpy rather than 100k JPEG decodes.

Batch numbering follows the convention the rest of this code base uses: batch
numbers are 1-based and batch ``b`` holds training images
``[(b-1)*batch_size, b*batch_size)``. With the default ``batch_size`` of 1000
that makes ``--train-range=1-100`` one full epoch and ``--test-range=1-10`` the
whole 10k validation set. The C++ side further slices each batch into
``minibatch_size`` chunks.
"""

import os

import numpy as n

from data import *


class TinyImageNetDataProvider(LabeledDataProvider):
    IMG_SIZE = 64
    NUM_COLORS = 3
    DEFAULT_BATCH_SIZE = 1000

    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None,
                 dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch,
                                     init_batchnum, dp_params, test)

        self.img_size = self.IMG_SIZE
        self.num_colors = self.NUM_COLORS
        self.data_mean = n.load(os.path.join(data_dir, 'tiny-imagenet-mean.npy')).astype(n.single)

        split = 'val' if test else 'train'
        self.images = n.load(os.path.join(data_dir, 'tiny-imagenet-%s.npy' % split), mmap_mode='r')
        self.labels = n.load(os.path.join(data_dir, 'tiny-imagenet-%s-labels.npy' % split))

        # Input scaling. The 0..255 convention is what the 2012 ImageNet
        # provider used; ALEXNET_DATA_SCALE=0.00392156862745098 (1/255) is
        # available for input-normalised training.
        self.data_scale = float(os.environ.get('ALEXNET_DATA_SCALE', '1.0'))
        self.batch_size = int(dp_params.get('batch_size', self.DEFAULT_BATCH_SIZE))
        self.num_classes = len(self.batch_meta['label_names'])
        self.data_dims = [self.IMG_SIZE ** 2 * self.NUM_COLORS, 1, self.num_classes]

    # -- metadata ---------------------------------------------------------

    def get_batch_meta(self, data_dir):
        """Called from DataProvider.__init__ before anything else is set up, so
        it must not touch instance state."""
        names = []
        path = os.path.join(data_dir, 'tiny-imagenet-classes.txt')
        if os.path.exists(path):
            with open(path) as fh:
                names = [line.split()[1] for line in fh if line.strip()]
        if not names:
            names = ['class%d' % i for i in range(200)]
        return {'label_names': names}

    def get_num_classes(self):
        return len(self.batch_meta['label_names'])

    def get_data_dims(self, idx=0):
        return self.data_dims[idx]

    # -- batching ---------------------------------------------------------

    def get_batch(self, batch_num):
        # Batch numbers are 1-based. Batch 0 (and negatives) appear when
        # --check-grads forces the batch range to "0"; treat those as batch 1 so
        # the gradient checker can pull a real minibatch.
        idx = max(batch_num, 1) - 1
        i0 = idx * self.batch_size
        if i0 >= self.images.shape[0]:
            raise IndexError('batch %d out of range for %d images (batch_size=%d)'
                             % (batch_num, self.images.shape[0], self.batch_size))
        i1 = min(i0 + self.batch_size, self.images.shape[0])

        # (n, 3, 64, 64) -> (n, 3*64*64) -> (3*64*64, n):
        # row index is c*H*W + y*W + x, the layout cudaconv2 expects.
        imgs = self.images[i0:i1]
        data = n.require(imgs.reshape(imgs.shape[0], -1).T, dtype=n.single, requirements='C')
        data *= self.data_scale
        data -= self.data_mean * self.data_scale
        return {'data': data, 'labels': self.labels[i0:i1].astype(n.int64)}

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledDataProvider.get_next_batch(self)

        data = datadic['data']
        labels = n.asarray(datadic['labels']).astype(n.int64)
        num_cases = data.shape[1]

        labels_vec = n.require(labels.reshape((1, num_cases)), dtype=n.single, requirements='C')
        labels_mat = n.zeros((self.num_classes, num_cases), dtype=n.single)
        labels_mat[labels, n.arange(num_cases)] = 1

        return epoch, batchnum, [data, labels_vec, labels_mat]

    # -- presentation -----------------------------------------------------

    def get_plottable_data(self, data):
        """(numPixels, numCases) -> (numCases, imgSize, imgSize, 3) for pylab."""
        num_cases = data.shape[1]
        return n.require((data + self.data_mean).T.reshape(
            num_cases, self.NUM_COLORS, self.IMG_SIZE, self.IMG_SIZE).swapaxes(1, 3).swapaxes(1, 2) / 255.0,
            dtype=n.single)
