#!/usr/bin/env python3
"""Convert a Tiny-ImageNet-200 tree into the flat .npy caches the AlexNet
port's data provider reads.

Input layout (the official cs231n zip):

    tiny-imagenet-200/
        train/<wnid>/images/<wnid>_<n>.JPEG      100000 images, 200 classes
        val/images/val_<n>.JPEG                   10000 images
        val/val_annotations.txt                   <file>\t<wnid>\t...
        wnids.txt                                 200 wnids, one per line

Output (written next to the input by default):

    tiny-imagenet-train.npy          uint8   (100000, 3, 64, 64)
    tiny-imagenet-train-labels.npy   int16   (100000,)
    tiny-imagenet-val.npy            uint8   (10000, 3, 64, 64)
    tiny-imagenet-val-labels.npy     int16   (10000,)
    tiny-imagenet-mean.npy           float32 (3*64*64, 1)   per-pixel mean
    tiny-imagenet-classes.txt        200 lines: "<label_index> <wnid>"

Decoding 110k JPEGs once and memory-mapping the result turns every later epoch
into a pure memcpy, which matters because the 2012 kernels are slow enough that
per-epoch JPEG decoding would dominate.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

IMG_SIZE = 64
NUM_COLORS = 3
NUM_CLASSES = 200


def decode(path):
    """Load one image as a (3, 64, 64) uint8 array in channel-major order."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert('RGB'), dtype=np.uint8)
    if arr.shape != (IMG_SIZE, IMG_SIZE, NUM_COLORS):
        raise ValueError('%s: unexpected shape %s' % (path, arr.shape))
    # (H, W, C) -> (C, H, W) so that a flattened pixel index is
    # c*H*W + y*W + x, which is the layout cudaconv2 expects.
    return arr.transpose(2, 0, 1)


def progress(done, total, label):
    pct = 100.0 * done / max(total, 1)
    sys.stdout.write('\r  %s: %d/%d (%.1f%%)' % (label, done, total, pct))
    sys.stdout.flush()


def build_split(root, wnids, split, out_dir):
    """Decode one split into (images, labels) arrays."""
    wnid_to_label = {w: i for i, w in enumerate(wnids)}

    if split == 'train':
        items = []
        for wnid in wnids:
            img_dir = os.path.join(root, 'train', wnid, 'images')
            for name in sorted(os.listdir(img_dir)):
                if name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    items.append((os.path.join(img_dir, name), wnid_to_label[wnid]))
    else:
        img_dir = os.path.join(root, 'val', 'images')
        ann = os.path.join(root, 'val', 'val_annotations.txt')
        with open(ann) as fh:
            mapping = {}
            for line in fh:
                parts = line.split('\t')
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        items = []
        for name in sorted(os.listdir(img_dir)):
            if name in mapping:
                items.append((os.path.join(img_dir, name), wnid_to_label[mapping[name]]))
            else:
                raise ValueError('no annotation for %s' % name)

    n_items = len(items)
    images = np.empty((n_items, NUM_COLORS, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    labels = np.empty((n_items,), dtype=np.int16)

    for i, (path, label) in enumerate(items):
        images[i] = decode(path)
        labels[i] = label
        if i % 500 == 0 or i == n_items - 1:
            progress(i + 1, n_items, split)
    sys.stdout.write('\n')

    np.save(os.path.join(out_dir, 'tiny-imagenet-%s.npy' % split), images)
    np.save(os.path.join(out_dir, 'tiny-imagenet-%s-labels.npy' % split), labels)
    print('  %s: %s images, %d classes present'
          % (split, n_items, len(np.unique(labels))))
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='path to the tiny-imagenet-200 directory')
    ap.add_argument('--out', default=None, help='output directory (default: the data dir)')
    ap.add_argument('--mean-samples', type=int, default=20000,
                    help='number of training images used for the per-pixel mean')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out) if args.out else root
    os.makedirs(out_dir, exist_ok=True)

    wnids_file = os.path.join(root, 'wnids.txt')
    with open(wnids_file) as fh:
        wnids = [line.strip() for line in fh if line.strip()]
    if len(wnids) != NUM_CLASSES:
        raise SystemExit('expected %d wnids in %s, found %d'
                         % (NUM_CLASSES, wnids_file, len(wnids)))

    print('class list: %d wnids' % len(wnids))
    train = build_split(root, wnids, 'train', out_dir)
    build_split(root, wnids, 'val', out_dir)

    # Per-pixel mean over a subsample of the training set, flattened to the
    # (numPixels, 1) shape the data provider subtracts.
    k = min(args.mean_samples, train.shape[0])
    step = max(train.shape[0] // k, 1)
    sample = train[::step][:k].astype(np.float32)
    mean = sample.mean(axis=0).reshape(-1, 1)
    np.save(os.path.join(out_dir, 'tiny-imagenet-mean.npy'), mean)
    print('mean image: computed from %d training images' % sample.shape[0])

    with open(os.path.join(out_dir, 'tiny-imagenet-classes.txt'), 'w') as fh:
        for i, w in enumerate(wnids):
            fh.write('%d %s\n' % (i, w))

    print('done -> %s' % out_dir)


if __name__ == '__main__':
    main()
