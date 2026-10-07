# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
import pickle
import pickle as _pickle   # the module-level names below shadow `pickle`
import os
import subprocess
import numpy as n
from math import sqrt

import gzip
import zipfile

class UnpickleError(Exception):
    pass

VENDOR_ID_REGEX = re.compile('^vendor_id\s+: (\S+)')
GPU_LOCK_NO_SCRIPT = -2
GPU_LOCK_NO_LOCK = -1

try:
    import magic
    ms = magic.open(magic.MAGIC_NONE)
    ms.load()
except ImportError: # no magic module
    ms = None

def get_gpu_lock(id=-1):
    import imp
    lock_script_path = '/u/tang/bin/gpu_lock2.py'
    if os.path.exists(lock_script_path):
        locker = imp.load_source("", lock_script_path)
        if id == -1:
            return locker.obtain_lock_id()
        print(id)
        got_id = locker._obtain_lock(id)
        return id if got_id else GPU_LOCK_NO_LOCK
    return GPU_LOCK_NO_SCRIPT if id < 0 else id

def pickle(filename, data, compress=False):
    if compress:
        fo = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        fo.writestr('data', _pickle.dumps(data, -1))
    else:
        fo = open(filename, "wb")
        _pickle.dump(data, fo, protocol=_pickle.HIGHEST_PROTOCOL)
    fo.close()
    
def unpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)
    if ms is not None and ms.file(filename).startswith('gzip'):
        fo = gzip.open(filename, 'rb')
        dict = _pickle.load(fo)
    elif ms is not None and ms.file(filename).startswith('Zip'):
        fo = zipfile.ZipFile(filename, 'r', zipfile.ZIP_DEFLATED)
        dict = _pickle.loads(fo.read('data'))
    else:
        fo = open(filename, 'rb')
        dict = _pickle.load(fo)
    
    fo.close()
    return dict

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def is_intel_machine():
    f = open('/proc/cpuinfo')
    for line in f:
        m = VENDOR_ID_REGEX.match(line)
        if m:
            f.close()
            return m.group(1) == 'GenuineIntel'
    f.close()
    return False

def get_cpu():
    if is_intel_machine():
        return 'intel'
    return 'amd'

def is_kepler_machine():
    """True when the installed GPU is a Kepler (compute capability 3.x) part.

    The private 2012 tree shipped two builds of the extension: a generic
    ``_ConvNet`` and a Kepler-tuned ``_ConvNet_k20x``; convnet.py picks between
    them with this predicate. It is not present in the public cuda-convnet
    drop, so it is implemented here by asking the driver for the compute
    capability. This port builds a single sm_70 extension, so the answer is
    False on the V100-class hardware it targets.
    """
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            stderr=subprocess.STDOUT)
    except Exception:
        return False
    first = out.decode('utf-8', 'replace').strip().splitlines()
    if not first:
        return False
    return first[0].strip().startswith('3.')

def is_windows_machine():
    return os.name == 'nt'

def get_device_cpus(device_id):
    """CPU ids that are NUMA-local to a given CUDA device.

    ConvNetGPU pins each GPU's worker thread to the cores attached to that
    GPU's PCIe root complex. This data is not computed anywhere in the public
    cuda-convnet drop, so it is derived here from nvidia-smi plus sysfs.
    Returns [] when the topology cannot be determined, in which case the C++
    Thread constructor simply skips affinity binding.
    """
    try:
        bdf = None
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'],
            stderr=subprocess.STDOUT)
        for line in out.decode('utf-8', 'replace').strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == device_id:
                bdf = parts[1].lower()
                break
        if bdf is None:
            return []
        # nvidia-smi prints an 8-hex-digit PCI domain, sysfs uses 4.
        if len(bdf) > 12:
            bdf = bdf[-12:]
        with open('/sys/bus/pci/devices/%s/numa_node' % bdf) as fh:
            node = int(fh.read().strip())
        if node < 0:
            return []
        with open('/sys/devices/system/node/node%d/cpulist' % node) as fh:
            cpulist = fh.read().strip()
        cpus = []
        for part in cpulist.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                lo, hi = part.split('-')
                cpus.extend(range(int(lo), int(hi) + 1))
            else:
                cpus.append(int(part))
        return cpus
    except Exception:
        return []
