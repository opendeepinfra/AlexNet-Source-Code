#include <quantizer.cuh>
#include <cutil_inline.h>
#include <cuda_fp16.h>

using namespace std;

/* ---------------------------------------------------------------------------
 * Half-precision quantization helpers.
 *
 * convQuantizeHalf() / convDequantizeHalf() live in the private "cudaconv2"
 * library of the original 2012 tree and are absent from the public
 * cuda-convnet drop, so they are reimplemented here.  They are used by
 * HalfQuantizer to move activations/gradients between devices as 16-bit
 * values (half the PCIe traffic).
 *
 * Semantics match Quantizer's float path:
 *   quantize   : tgt[i] = (half) src[i]
 *   dequantize : tgt[i] = scaleTarget * tgt[i] + scaleOutput * (float) src[i]
 * ------------------------------------------------------------------------- */
static const int HALF_QUANT_BLOCK = 512;

__global__ void kQuantizeHalf(const float *src, __half *tgt, int numEls) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numEls) {
        tgt[i] = __float2half_rn(src[i]);
    }
}

__global__ void kDequantizeHalf(const __half *src, float *tgt, int numEls,
                                float scaleTarget, float scaleOutput) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numEls) {
        tgt[i] = scaleTarget * tgt[i] + scaleOutput * __half2float(src[i]);
    }
}

void convQuantizeHalf(NVMatrix &src, NVMatrix &tgt) {
    int numEls = (int) src.getNumElements();
    if (numEls <= 0) {
        return;
    }
    int blocks = (numEls + HALF_QUANT_BLOCK - 1) / HALF_QUANT_BLOCK;
    kQuantizeHalf<<<blocks, HALF_QUANT_BLOCK>>>(
        (const float *) src.getDevData(), (__half *) tgt.getDevData(), numEls);
    cutilCheckMsg("convQuantizeHalf failed");
}

void convDequantizeHalf(NVMatrix &src, NVMatrix &tgt, int numEls,
                        float scaleTarget, float scaleOutput) {
    if (numEls <= 0) {
        return;
    }
    int blocks = (numEls + HALF_QUANT_BLOCK - 1) / HALF_QUANT_BLOCK;
    kDequantizeHalf<<<blocks, HALF_QUANT_BLOCK>>>(
        (const __half *) src.getDevData(), tgt.getDevData(), numEls,
        scaleTarget, scaleOutput);
    cutilCheckMsg("convDequantizeHalf failed");
}

/*=================
 * Quantizer
 * ================
 */

Quantizer& Quantizer::make(PyObject* lrsDict) {
    string type = pyDictGetString(lrsDict, "type");
    if (type == "default") {
        return *new Quantizer();
    } else if (type == "half") {
        return *new HalfQuantizer();
    }
    throw string("Unknown quantizer type ") + type;
}

Quantizer::Quantizer() : _numRows(0), _numCols(0), _trans(false) {
}

Quantizer::~Quantizer() {
}

void Quantizer::quantize(NVMatrix& src, NVMatrix& tgt) {
    _quantize(src, tgt);
    _quantized = &tgt;
    _numRows = src.getNumRows();
    _numCols = src.getNumCols();
    _trans = src.isTrans();
}

void Quantizer::dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput) {
    _dequantize(tgt, scaleTarget, scaleOutput);
    tgt.setTrans(_trans);
    tgt.reshape(_numRows, _numCols);
}

void Quantizer::dequantize(NVMatrix& tgt) {
    dequantize(tgt, 0, 1);
}

void Quantizer::_quantize(NVMatrix& src, NVMatrix& tgt) {
    src.copy(tgt);
}

void Quantizer::_dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput) {
    tgt.add(*_quantized, scaleTarget, scaleOutput);
}

/*=================
 * HalfQuantizer
 * ================
 */
HalfQuantizer::HalfQuantizer() : Quantizer() {
}

void HalfQuantizer::_quantize(NVMatrix& src, NVMatrix& tgt) {
    convQuantizeHalf(src, tgt);
}

void HalfQuantizer::_dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput) {
    convDequantizeHalf(*_quantized, tgt, _numRows * _numCols, scaleTarget, scaleOutput);
}
