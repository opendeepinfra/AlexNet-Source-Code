# Running the 2012 AlexNet source on a modern GPU / K8S platform

This fork adds a port of the original 2012 AlexNet (cuda-convnet) source so that
it builds and runs on a modern GPU (Tesla V100, `sm_70`) and can be submitted as
a task on a Kubernetes "Hai Platform" cluster.

The upstream release is a *partial* source drop: it is the private 2012 tree of
Krizhevsky's cuda-convnet with the shared infrastructure libraries removed.
Nothing here changes the model or the training algorithm; the work is build
system, API reconstruction and Python 2 → 3 migration.

---

## 1. What was missing from the release

| Referenced by | Missing | Resolution |
| --- | --- | --- |
| `src/*.cu`, `include/*.cuh` | `nvmatrix` (matrix library) | taken from the public cuda-convnet drop |
| `src/layer.cu` | `cudaconv2` (conv kernels) | taken from the public cuda-convnet drop |
| `include/*.cuh` | `matrix.h`, `queue.h`, `thread.h`, `sync.h` (the author's "mycpp" utilities) | `matrix/queue/thread` from public cuda-convnet, `sync.h` from cuda-convnet2 |
| 11 files | `helper_cuda.h`, `cutil_inline.h`, `helper_timer.h` (CUDA SDK headers) | reconstructed minimally in `include/common/` |
| `src/common/matrix.cpp` | `cblas.h` (ATLAS) | replaced by a self-contained `include/common/cblas.h` — removes the BLAS runtime dependency |
| `convnet.py`, `layer.py` | `util.py`, `options.py`, `data.py`, `gpumodel.py`, `ordereddict.py` | taken from the public cuda-convnet drop |
| `Makefile-distrib` | `common-gcc-cuda-4.0.mk` | not needed; replaced by `Makefile-k8s` |

`package.sh` and `build.sh` document the original assembly steps and confirm the
provenance of every file above.

## 2. Reconstructed APIs

These symbols are used by the release but exist in *no* public cuda-convnet
version (they were added in the private 2012 branch). They were reimplemented:

| Symbol | Where | Note |
| --- | --- | --- |
| `NVMatrix::setDeviceID(int)` | `src/nvmatrix/nvmatrix.cu` | counterpart of the existing `getDeviceID()` |
| `NVMatrix::canAccessDevice(int,int)` | `src/nvmatrix/nvmatrix.cu` | `cudaDeviceCanAccessPeer`, true for same device |
| `NVMatrix::initCublas()` | `src/nvmatrix/nvmatrix.cu` | explicit `cublasInit()` |
| `NVMatrix::getClone()` | `src/nvmatrix/nvmatrix.cu` | identical to the existing `copy()` |
| `HostNVMatrix` | `include/nvmatrix/nvmatrix.cuh` | cross-device staging matrix (unused on 1 GPU) |
| `convReflectHorizontal` | `src/cudaconv2/conv_util.cu` | ported from cuda-convnet2 `cudaconv3` |
| `convLocalRandomPool` | `src/cudaconv2/conv_util.cu` | no public implementation exists; written to match `RandomPoolLayer` |
| `Thread(bool, std::vector<int>&)` | `include/common/thread.h` | CPU-affinity constructor used by `ConvNetGPU` |
| `minDiv` parameter | `src/cudaconv2/conv_util.cu` | the newer cross-map response norm exposes the `k` term that the public version hard-coded as `1` |

## 3. Build

```sh
make -f Makefile-k8s -j"$(nproc)"
```

Targets CUDA 11.8/12.x, `sm_70` (V100) and Python 3.8, and produces
`_ConvNet.so`, the CPython extension the Python code imports.

Changes relative to the original build:

* `-gencode arch=compute_20` → `compute_70` (`sm_20` is long unsupported);
* `src/test.cu` excluded — it is a standalone `main()` unit test that needs the
  CUDA SDK timer and a `test.cuh` that is not part of the release;
* numpy's include directory is `.../numpy/core/include/numpy` (the sources use
  `#include <arrayobject.h>` without the `numpy/` prefix);
* `cutilCheckMsg` / `checkCudaErrors` / `getLastCudaError` provided locally.

## 4. Python 2 → 3

`lib2to3` was applied, followed by targeted fixes:

* `src/util.cu`, `src/convnet.cu`: `PyString_*`→`PyUnicode_*`, `PyInt_*`→`PyLong_*`;
* `src/pyconvnet.cu`: `Py_InitModule` → `PyModule_Create` and
  `init_ConvNet` → `PyInit__ConvNet` (the module initialiser Python 3 requires).

## 5. Licence / provenance

The AlexNet sources keep their original BSD headers. The files imported from the
public cuda-convnet drop carry Krizhevsky's BSD header; `include/common/sync.h`
is from cuda-convnet2 and carries its Apache-2.0 header. Those headers are
preserved verbatim.

---

## 6. Running on a Kubernetes "Hai Platform" (Tiny-ImageNet)

`alexnet_tinyimagenet.yml` + `run_tinyimagenet.py` submit a training run of this
same 2012 code as a platform task. Nothing about the model is replaced — the
task runs `convnet.py` with the original layer parser, the original CUDA
kernels and the original training loop.

### 6.1 What was added

| File | Purpose |
| --- | --- |
| `make-data/prepare_tinyimagenet.py` | decodes the official Tiny-ImageNet-200 tree once into flat `.npy` caches (+ per-pixel mean) |
| `convdata_tinyimagenet.py` | data provider serving those caches; batch `b` = images `[(b-1)*1000, b*1000)`, memory-mapped |
| `layers-tinyimagenet/*.cfg` | AlexNet topology adapted to 64x64 input and a **single** GPU |
| `run_tinyimagenet.py` | task entrypoint: preloads the CUDA runtime libs shipped in `pylibs/lib/`, then runs `convnet.py` |
| `alexnet_tinyimagenet.yml` | the task spec (`hai-cli run alexnet_tinyimagenet.yml`) |

The task image is a plain Python 3.8 runtime with no CUDA toolkit, so
`_ConvNet.so` cannot resolve `libcudart.so.12` / `libcublas.so.12` on its own.
Those two libraries (plus `libcublasLt.so.12`) are placed in `pylibs/lib/` and
loaded with `ctypes.CDLL(..., RTLD_GLOBAL)` before the extension is imported.

### 6.2 Layer definition

`layers/layers-120.cfg` is the 2012 ImageNet definition: two columns split over
two GPUs that only exchange data at conv3 and the first FC layer, over 224x224
crops. `layers-tinyimagenet/layers-tinyimagenet.cfg` keeps the topology (5
convolutions + ReLU, max pooling, cross-map response normalisation after conv1
and conv2, three FC layers with dropout on the last two) but:

* runs one column on `gpu=0`, so no cross-device copies happen;
* fits 64x64 input (64 -> 32 -> 16 -> 8 feature maps);
* widens conv1 to 64 filters, matching the 2-GPU net's 32+32.

### 6.3 The learning rate depends on the input scale

This is the one finding that matters operationally, and it is **not** a porting
bug.

The 2012 ImageNet provider feeds pixels on a 0..255 scale, and the ImageNet
learning rates in `layers/layer-params-120-2012.cfg` (`epsW` 0.01 -> 0.00001)
were tuned for that. Because weight gradients scale with the magnitude of the
layer's inputs, feeding 0..255 pixels makes the effective step ~255x larger than
the same `epsW` on [0,1] inputs.

Measured on this V100 with the 64x64 Tiny-ImageNet net and 0..255 inputs:

| `epsW` | behaviour |
| --- | --- |
| 0.001 | diverges: cost 5.2 -> 10.9 within two minibatches, top-5 error -> 100% |
| 0.0003 | oscillates around 5.0-6.0 |
| 0.0001 | marginally stable; epoch 1 rises, later epochs creep down (5.15 -> 4.99 over 3 epochs) |
| 0.00001 | stable but very slow |

Scaling the inputs to [0,1] (`ALEXNET_DATA_SCALE=0.00392156862745098`, set by
`convdata_tinyimagenet.py` from the environment) restores the intended
behaviour and the **original** `epsW=0.01` becomes stable and learns:

```
normalised inputs, epsW=0.01, 20 batches/epoch:
  epoch 2 -> 5.50   epoch 3 -> 5.03   ... epoch 7 -> 4.21
```

So the shipped configuration normalises the input rather than re-tuning the
2012 learning rates.

### 6.4 Platform notes

* `--gpu 0` must be passed explicitly: `get_gpu_lock(-1)` returns the
  `GPU_LOCK_NO_SCRIPT` sentinel when the private `gpu_lock2.py` is absent, and
  that sentinel is an invalid CUDA ordinal.
* `get_device_cpus()` (in `util.py`) derives NVLink/PCIe-local CPU sets from
  `nvidia-smi` + sysfs for the CPU-affinity constructor of `Thread`; it returns
  an empty list when the topology is unavailable.
* `util.pickle()` shadowed the `pickle` module it was calling, so saving a
  checkpoint raised `AttributeError: 'function' object has no attribute 'dump'`
  on Python 3. The stdlib module is now imported as `_pickle`.
* Two `sorted()` calls relied on Python 2's implicit ordering of arbitrary
  objects, and `options.py` passed `cmp=` to `sorted()`; both are Python 3
  errors now fixed with explicit sort keys / `functools.cmp_to_key`.
* `layer.py` had ~15 integer divisions (`imgPixels`, `filterChannels`,
  `outputsX`, Gaussian filter centre, ...) that Python 3 turns into float
  division; all are now `//`.
