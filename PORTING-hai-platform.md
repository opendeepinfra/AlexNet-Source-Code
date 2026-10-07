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
