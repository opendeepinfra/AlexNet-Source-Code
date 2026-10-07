#!/usr/bin/env python3
"""Hai Platform entrypoint for the ported AlexNet (Tiny-ImageNet).

The platform task image is a plain Python 3.8 runtime with no CUDA toolkit, so
``_ConvNet.so`` cannot resolve its ``libcudart.so.12`` / ``libcublas.so.12``
dependencies by itself. Those shared objects are shipped in
``pylibs/lib/`` next to this file and are loaded here with ``ctypes`` and
``RTLD_GLOBAL`` *before* the extension is imported; the dynamic loader then
satisfies the extension's NEEDED entries from the already-loaded sonames.

This is the same technique the platform's own examples use for native
extensions that need libraries the base image does not carry.

Everything below just forwards the platform-provided ``parameters`` to
convnet.py, so the task spec stays a thin wrapper around the original program.
"""

import ctypes
import os
import runpy
import sys

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
LIBDIR = os.path.join(WORKSPACE, 'pylibs', 'lib')

# Order matters: libcublas depends on libcublasLt.
CUDA_LIBS = ('libcublasLt.so.12', 'libcublas.so.12', 'libcudart.so.12')


def preload_cuda_libraries():
    loaded = []
    for name in CUDA_LIBS:
        path = os.path.join(LIBDIR, name)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded.append(name)
    if loaded:
        print('[run_tinyimagenet] preloaded CUDA libraries: %s' % ', '.join(loaded))
    else:
        print('[run_tinyimagenet] WARNING: no CUDA libraries found in %s; '
              'expecting them to come from the system' % LIBDIR)


def main():
    preload_cuda_libraries()

    os.chdir(WORKSPACE)
    sys.path.insert(0, WORKSPACE)

    convnet_py = os.path.join(WORKSPACE, 'convnet.py')
    sys.argv = [convnet_py] + sys.argv[1:]
    runpy.run_path(convnet_py, run_name='__main__')


if __name__ == '__main__':
    main()
