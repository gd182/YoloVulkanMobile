#include "qnn_yolo.h"

#include <android/log.h>

#define QLOG_TAG "QnnYolo"
#define QLOGI(...) __android_log_print(ANDROID_LOG_INFO, QLOG_TAG, __VA_ARGS__)
#define QLOGE(...) __android_log_print(ANDROID_LOG_ERROR, QLOG_TAG, __VA_ARGS__)

#if defined(HAVE_QNN)

#include <dlfcn.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <mat.h>

#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

namespace
{

using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
using GetSystemProvidersFn = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

Qnn_DataType_t tensorType(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType; }
uint32_t tensorRank(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank; }
const uint32_t* tensorDims(const Qnn_Tensor_t& t) { return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions; }
const Qnn_QuantizeParams_t& tensorQuant(const Qnn_Tensor_t& t)
{
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.quantizeParams : t.v1.quantizeParams;
}

void setClientBuffer(Qnn_Tensor_t& t, void* data, uint32_t size)
{
    if (t.version == QNN_TENSOR_VERSION_2)
    {
        t.v2.memType = QNN_TENSORMEMTYPE_RAW;
        t.v2.clientBuf.data = data;
        t.v2.clientBuf.dataSize = size;
    }
    else
    {
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf.data = data;
        t.v1.clientBuf.dataSize = size;
    }
}

size_t dtypeSize(Qnn_DataType_t t)
{
    switch (t)
    {
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
        return 4;
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
        return 2;
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
        return 1;
    default:
        return 0;
    }
}

const char* dtypeName(Qnn_DataType_t t)
{
    switch (t)
    {
    case QNN_DATATYPE_FLOAT_32: return "fp32";
    case QNN_DATATYPE_FLOAT_16: return "fp16";
    case QNN_DATATYPE_UFIXED_POINT_8: return "u8q";
    case QNN_DATATYPE_UFIXED_POINT_16: return "u16q";
    case QNN_DATATYPE_SFIXED_POINT_8: return "i8q";
    case QNN_DATATYPE_SFIXED_POINT_16: return "i16q";
    case QNN_DATATYPE_UINT_8: return "u8";
    case QNN_DATATYPE_INT_8: return "i8";
    default: return "other";
    }
}

struct Quant
{
    float scale = 1.f;
    int32_t offset = 0;
};

Quant quantOf(const Qnn_Tensor_t& t)
{
    Quant q;
    const Qnn_QuantizeParams_t& p = tensorQuant(t);
    if (p.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
    {
        q.scale = p.scaleOffsetEncoding.scale;
        q.offset = p.scaleOffsetEncoding.offset;
    }
    else if (p.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET)
    {
        q.scale = p.bwScaleOffsetEncoding.scale;
        q.offset = p.bwScaleOffsetEncoding.offset;
    }
    return q;
}

float readElement(const uint8_t* base, size_t index, Qnn_DataType_t type, const Quant& q)
{
    switch (type)
    {
    case QNN_DATATYPE_FLOAT_32:
    {
        float v;
        std::memcpy(&v, base + index * 4, 4);
        return v;
    }
    case QNN_DATATYPE_FLOAT_16:
    {
        __fp16 v;
        std::memcpy(&v, base + index * 2, 2);
        return (float)v;
    }
    case QNN_DATATYPE_UFIXED_POINT_8:
        return ((int)base[index] + q.offset) * q.scale;
    case QNN_DATATYPE_UFIXED_POINT_16:
    {
        uint16_t v;
        std::memcpy(&v, base + index * 2, 2);
        return ((int)v + q.offset) * q.scale;
    }
    case QNN_DATATYPE_SFIXED_POINT_8:
        return ((int)(int8_t)base[index] + q.offset) * q.scale;
    case QNN_DATATYPE_SFIXED_POINT_16:
    {
        int16_t v;
        std::memcpy(&v, base + index * 2, 2);
        return ((int)v + q.offset) * q.scale;
    }
    case QNN_DATATYPE_UINT_8:
        return (float)base[index];
    case QNN_DATATYPE_INT_8:
        return (float)(int8_t)base[index];
    default:
        return 0.f;
    }
}

uint8_t quantize8(float v, const Quant& q, bool isSigned)
{
    int r = (int)std::lround(v / q.scale) - q.offset;
    if (isSigned) return (uint8_t)(int8_t)std::max(-128, std::min(127, r));
    return (uint8_t)std::max(0, std::min(255, r));
}

void writeElement(uint8_t* base, size_t index, Qnn_DataType_t type, const Quant& q, float v)
{
    switch (type)
    {
    case QNN_DATATYPE_FLOAT_32:
        std::memcpy(base + index * 4, &v, 4);
        break;
    case QNN_DATATYPE_FLOAT_16:
    {
        __fp16 h = (__fp16)v;
        std::memcpy(base + index * 2, &h, 2);
        break;
    }
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_UINT_8:
        base[index] = quantize8(v, q, false);
        break;
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_INT_8:
        base[index] = quantize8(v, q, true);
        break;
    case QNN_DATATYPE_UFIXED_POINT_16:
    {
        int r = (int)std::lround(v / q.scale) - q.offset;
        uint16_t u = (uint16_t)std::max(0, std::min(65535, r));
        std::memcpy(base + index * 2, &u, 2);
        break;
    }
    case QNN_DATATYPE_SFIXED_POINT_16:
    {
        int r = (int)std::lround(v / q.scale) - q.offset;
        int16_t s = (int16_t)std::max(-32768, std::min(32767, r));
        std::memcpy(base + index * 2, &s, 2);
        break;
    }
    default:
        break;
    }
}

bool supportedType(Qnn_DataType_t t) { return dtypeSize(t) != 0; }

std::string dimsToString(const uint32_t* d, uint32_t rank)
{
    std::string s = "[";
    for (uint32_t i = 0; i < rank; i++)
    {
        if (i) s += ",";
        s += std::to_string(d[i]);
    }
    return s + "]";
}

float iouOf(const Object& a, const Object& b)
{
    float x0 = std::max(a.x, b.x);
    float y0 = std::max(a.y, b.y);
    float x1 = std::min(a.x + a.w, b.x + b.w);
    float y1 = std::min(a.y + a.h, b.y + b.h);
    float inter = std::max(0.f, x1 - x0) * std::max(0.f, y1 - y0);
    float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

}

struct QnnYolo::Impl
{
    void* backendLib = nullptr;
    void* systemLib = nullptr;
    QNN_INTERFACE_VER_TYPE qnn{};
    QNN_SYSTEM_INTERFACE_VER_TYPE sys{};
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_GraphHandle_t graph = nullptr;
    QnnSystemContext_Handle_t sysCtx = nullptr;

    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
    std::vector<std::vector<uint8_t>> inputBufs;
    std::vector<std::vector<uint8_t>> outputBufs;

    int inW = 0;
    int inH = 0;
    bool nhwc = true;
    int numClass = 0;
    bool boxesNormalized = false;
    bool channelFirst = true;
    bool splitOutput = false;
    bool boxChannelFirst = true;
    bool scoreChannelFirst = true;
    int channels = 0;
    int anchors = 0;
    std::string info;

    std::vector<uint8_t> lut;
    std::vector<float> outF;
    std::vector<Object> proposals;
    std::vector<Object> kept;
    double accPre = 0;
    double accExec = 0;
    double accPost = 0;
    int accFrames = 0;
    bool inputRangeLogged = false;
    bool outputRangeLogged = false;

    ~Impl() { release(); }

    void release()
    {
        if (sysCtx && sys.systemContextFree) sys.systemContextFree(sysCtx);
        sysCtx = nullptr;
        if (context && qnn.contextFree) qnn.contextFree(context, nullptr);
        context = nullptr;
        if (device && qnn.deviceFree) qnn.deviceFree(device);
        device = nullptr;
        if (backend && qnn.backendFree) qnn.backendFree(backend);
        backend = nullptr;
        inputs.clear();
        outputs.clear();
        if (systemLib) dlclose(systemLib);
        systemLib = nullptr;
        if (backendLib) dlclose(backendLib);
        backendLib = nullptr;
    }

    std::string open(const std::string& nativeLibDir)
    {
        std::string adsp = nativeLibDir + ";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp";
        setenv("ADSP_LIBRARY_PATH", adsp.c_str(), 1);

        backendLib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
        if (!backendLib) return std::string("dlopen libQnnHtp.so failed: ") + dlerror();
        systemLib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
        if (!systemLib) return std::string("dlopen libQnnSystem.so failed: ") + dlerror();

        auto getProviders = (GetProvidersFn)dlsym(backendLib, "QnnInterface_getProviders");
        if (!getProviders) return "QnnInterface_getProviders not found";
        const QnnInterface_t** providers = nullptr;
        uint32_t count = 0;
        if (getProviders(&providers, &count) != QNN_SUCCESS || count == 0) return "no QNN backend providers";
        bool found = false;
        for (uint32_t i = 0; i < count; i++)
        {
            const Qnn_Version_t& v = providers[i]->apiVersion.coreApiVersion;
            if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR)
            {
                qnn = providers[i]->QNN_INTERFACE_VER_NAME;
                found = true;
                break;
            }
        }
        if (!found) return "no compatible QNN backend API version (SDK headers vs runtime libs mismatch)";

        auto getSysProviders = (GetSystemProvidersFn)dlsym(systemLib, "QnnSystemInterface_getProviders");
        if (!getSysProviders) return "QnnSystemInterface_getProviders not found";
        const QnnSystemInterface_t** sysProviders = nullptr;
        uint32_t sysCount = 0;
        if (getSysProviders(&sysProviders, &sysCount) != QNN_SUCCESS || sysCount == 0) return "no QNN system providers";
        found = false;
        for (uint32_t i = 0; i < sysCount; i++)
        {
            const Qnn_Version_t& v = sysProviders[i]->systemApiVersion;
            if (v.major == QNN_SYSTEM_API_VERSION_MAJOR && v.minor >= QNN_SYSTEM_API_VERSION_MINOR)
            {
                sys = sysProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
                found = true;
                break;
            }
        }
        if (!found) return "no compatible QNN system API version";
        return "";
    }

    std::string createContext(const std::vector<uint8_t>& binary)
    {
        Qnn_ErrorHandle_t err = qnn.backendCreate(nullptr, nullptr, &backend);
        if (err != QNN_SUCCESS) return "backendCreate failed: " + std::to_string((long long)err);

        if (qnn.deviceCreate)
        {
            err = qnn.deviceCreate(nullptr, nullptr, &device);
            if (err != QNN_SUCCESS)
            {
                QLOGE("deviceCreate failed: %lld", (long long)err);
                device = nullptr;
            }
        }

        err = qnn.contextCreateFromBinary(backend, device, nullptr, binary.data(), binary.size(), &context, nullptr);
        if (err != QNN_SUCCESS)
            return "contextCreateFromBinary failed: " + std::to_string((long long)err) +
                   " (context binary must be built for this SoC's HTP arch with the same QNN SDK version as the runtime libs)";

        err = sys.systemContextCreate(&sysCtx);
        if (err != QNN_SUCCESS) return "systemContextCreate failed: " + std::to_string((long long)err);

        const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
        Qnn_ContextBinarySize_t binaryInfoSize = 0;
        err = sys.systemContextGetBinaryInfo(sysCtx, (void*)binary.data(), binary.size(), &binaryInfo, &binaryInfoSize);
        if (err != QNN_SUCCESS || !binaryInfo) return "systemContextGetBinaryInfo failed: " + std::to_string((long long)err);

        const QnnSystemContext_GraphInfo_t* graphs = nullptr;
        uint32_t numGraphs = 0;
        if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1)
        {
            graphs = binaryInfo->contextBinaryInfoV1.graphs;
            numGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
        }
        else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
        {
            graphs = binaryInfo->contextBinaryInfoV2.graphs;
            numGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
        }
        else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)
        {
            graphs = binaryInfo->contextBinaryInfoV3.graphs;
            numGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
        }
        else
        {
            return "unsupported context binary info version " + std::to_string((int)binaryInfo->version);
        }
        if (numGraphs == 0 || !graphs) return "context binary has no graphs";

        const char* graphName = nullptr;
        Qnn_Tensor_t* graphInputs = nullptr;
        Qnn_Tensor_t* graphOutputs = nullptr;
        uint32_t numIn = 0;
        uint32_t numOut = 0;
        if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1)
        {
            const auto& g = graphs[0].graphInfoV1;
            graphName = g.graphName;
            graphInputs = g.graphInputs;
            numIn = g.numGraphInputs;
            graphOutputs = g.graphOutputs;
            numOut = g.numGraphOutputs;
        }
        else if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2)
        {
            const auto& g = graphs[0].graphInfoV2;
            graphName = g.graphName;
            graphInputs = g.graphInputs;
            numIn = g.numGraphInputs;
            graphOutputs = g.graphOutputs;
            numOut = g.numGraphOutputs;
        }
        else if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3)
        {
            const auto& g = graphs[0].graphInfoV3;
            graphName = g.graphName;
            graphInputs = g.graphInputs;
            numIn = g.numGraphInputs;
            graphOutputs = g.graphOutputs;
            numOut = g.numGraphOutputs;
        }
        else
        {
            return "unsupported graph info version " + std::to_string((int)graphs[0].version);
        }
        if (numIn == 0 || numOut == 0) return "graph has no inputs or outputs";

        err = qnn.graphRetrieve(context, graphName, &graph);
        if (err != QNN_SUCCESS) return "graphRetrieve failed: " + std::to_string((long long)err);

        inputs.assign(graphInputs, graphInputs + numIn);
        outputs.assign(graphOutputs, graphOutputs + numOut);
        inputBufs.resize(numIn);
        outputBufs.resize(numOut);

        for (uint32_t i = 0; i < numIn; i++)
        {
            Qnn_DataType_t t = tensorType(inputs[i]);
            if (!supportedType(t)) return std::string("unsupported input tensor type ") + dtypeName(t);
            size_t n = 1;
            for (uint32_t d = 0; d < tensorRank(inputs[i]); d++) n *= tensorDims(inputs[i])[d];
            inputBufs[i].assign(n * dtypeSize(t), 0);
            setClientBuffer(inputs[i], inputBufs[i].data(), (uint32_t)inputBufs[i].size());
        }
        for (uint32_t i = 0; i < numOut; i++)
        {
            Qnn_DataType_t t = tensorType(outputs[i]);
            if (!supportedType(t)) return std::string("unsupported output tensor type ") + dtypeName(t);
            size_t n = 1;
            for (uint32_t d = 0; d < tensorRank(outputs[i]); d++) n *= tensorDims(outputs[i])[d];
            outputBufs[i].assign(n * dtypeSize(t), 0);
            setClientBuffer(outputs[i], outputBufs[i].data(), (uint32_t)outputBufs[i].size());
        }
        return "";
    }

    std::string configure()
    {
        const Qnn_Tensor_t& in = inputs[0];
        uint32_t rank = tensorRank(in);
        const uint32_t* d = tensorDims(in);
        if (rank != 4) return "input rank " + std::to_string(rank) + " is not 4";
        if (d[3] == 3)
        {
            nhwc = true;
            inH = (int)d[1];
            inW = (int)d[2];
        }
        else if (d[1] == 3)
        {
            nhwc = false;
            inH = (int)d[2];
            inW = (int)d[3];
        }
        else
        {
            return "input dims " + dimsToString(d, rank) + " are neither NHWC nor NCHW with 3 channels";
        }

        const Qnn_Tensor_t& out = outputs[0];
        uint32_t orank = tensorRank(out);
        const uint32_t* od = tensorDims(out);
        if (orank != 3) return "output rank " + std::to_string(orank) + " is not 3, dims " + dimsToString(od, orank);
        uint32_t expected = 4 + (uint32_t)numClass;
        if (outputs.size() == 2)
        {
            const uint32_t* sd = tensorDims(outputs[1]);
            if (tensorRank(outputs[1]) != 3)
                return "score output rank is not 3";
            if (od[1] == 4) { boxChannelFirst = true; anchors = (int)od[2]; }
            else if (od[2] == 4) { boxChannelFirst = false; anchors = (int)od[1]; }
            else return "box output dims " + dimsToString(od, orank) + " do not contain 4 channels";
            if (sd[1] == (uint32_t)numClass) { scoreChannelFirst = true; }
            else if (sd[2] == (uint32_t)numClass) { scoreChannelFirst = false; }
            else return "score output dims " + dimsToString(sd, 3) + " do not match " + std::to_string(numClass) + " classes";
            const int scoreAnchors = scoreChannelFirst ? (int)sd[2] : (int)sd[1];
            if (scoreAnchors != anchors) return "box and score outputs have different anchor counts";
            splitOutput = true;
            channelFirst = true;
            channels = (int)expected;
        }
        else if (od[1] == expected)
        {
            channelFirst = true;
            channels = (int)od[1];
            anchors = (int)od[2];
        }
        else if (od[2] == expected)
        {
            channelFirst = false;
            channels = (int)od[2];
            anchors = (int)od[1];
        }
        else
        {
            return "output dims " + dimsToString(od, orank) + " do not match 4+" + std::to_string(numClass) + " classes";
        }
        outF.assign((size_t)channels * anchors, 0.f);

        Qnn_DataType_t t = tensorType(in);
        Quant q = quantOf(in);
        const size_t lutEsz = dtypeSize(t);
        lut.assign(256 * lutEsz, 0);
        for (int p = 0; p < 256; p++)
            writeElement(lut.data(), (size_t)p, t, q, p / 255.f);

        const Qnn_Tensor_t& scoreOut = splitOutput ? outputs[1] : out;
        const Quant oq = quantOf(scoreOut);
        const Qnn_DataType_t outType = tensorType(scoreOut);
        const bool quantizedOutput = outType == QNN_DATATYPE_UFIXED_POINT_8 ||
                                     outType == QNN_DATATYPE_SFIXED_POINT_8 ||
                                     outType == QNN_DATATYPE_UFIXED_POINT_16 ||
                                     outType == QNN_DATATYPE_SFIXED_POINT_16;
        if (quantizedOutput && oq.scale > 0.25f)
        {
            return "quantized output scale " + std::to_string(oq.scale) +
                   " is too coarse for YOLO class scores (0..1); rebuild INT8 model with a float output";
        }
        info = "in " + dimsToString(d, rank) + " " + dtypeName(t) +
               " q(enc=" + std::to_string((int)tensorQuant(in).quantizationEncoding) +
               ",scale=" + std::to_string(q.scale) + ",offset=" + std::to_string(q.offset) + ") | out " +
               (splitOutput ? "split " : "") + dimsToString(od, orank) + " " + dtypeName(outType) +
               " q(enc=" + std::to_string((int)tensorQuant(scoreOut).quantizationEncoding) +
               ",scale=" + std::to_string(oq.scale) + ",offset=" + std::to_string(oq.offset) + ")";
        return "";
    }
};

QnnYolo::QnnYolo() : impl_(new Impl) {}

QnnYolo::~QnnYolo() = default;

std::string QnnYolo::describe() const { return impl_->info; }

std::string QnnYolo::load(AAssetManager* mgr, const std::string& model_asset, const std::string& native_lib_dir,
                          int num_class, bool boxes_normalized)
{
    AAsset* asset = AAssetManager_open(mgr, model_asset.c_str(), AASSET_MODE_STREAMING);
    if (!asset) return "cannot open asset " + model_asset;
    off_t length = AAsset_getLength(asset);
    std::vector<uint8_t> binary((size_t)length);
    size_t got = 0;
    while (got < binary.size())
    {
        int n = AAsset_read(asset, binary.data() + got, binary.size() - got);
        if (n <= 0) break;
        got += (size_t)n;
    }
    AAsset_close(asset);
    if (got != binary.size()) return "short read of asset " + model_asset;

    impl_->numClass = num_class;
    impl_->boxesNormalized = boxes_normalized;

    std::string err = impl_->open(native_lib_dir);
    if (!err.empty()) return err;
    err = impl_->createContext(binary);
    if (!err.empty()) return err;
    err = impl_->configure();
    if (!err.empty()) return err;
    QLOGI("ready: %s", impl_->info.c_str());
    return "";
}

int QnnYolo::detect(const unsigned char* rgba, int width, int height, std::vector<Object>& objects,
                    float prob_threshold, float nms_threshold)
{
    Impl& s = *impl_;
    objects.clear();
    if (!s.graph) return -1;
    const auto tStart = std::chrono::steady_clock::now();

    float scale = std::min((float)s.inW / width, (float)s.inH / height);
    int newW = (int)std::lround(width * scale);
    int newH = (int)std::lround(height * scale);
    int padX = (s.inW - newW) / 2;
    int padY = (s.inH - newH) / 2;

    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, width, height, newW, newH);
    ncnn::Mat padded;
    ncnn::copy_make_border(resized, padded, padY, s.inH - newH - padY, padX, s.inW - newW - padX,
                           ncnn::BORDER_CONSTANT, 114.f);

    const Qnn_Tensor_t& in = s.inputs[0];
    Qnn_DataType_t itype = tensorType(in);
    uint8_t* dst = s.inputBufs[0].data();
    const size_t esz = dtypeSize(itype);

    const uint8_t* lut = s.lut.data();
    auto fill = [&](auto store) {
        for (int c = 0; c < 3; c++)
        {
            const float* plane = padded.channel(c);
            for (int y = 0; y < s.inH; y++)
            {
                const float* row = plane + (size_t)y * s.inW;
                for (int x = 0; x < s.inW; x++)
                {
                    int p = (int)(row[x] + 0.5f);
                    p = p < 0 ? 0 : (p > 255 ? 255 : p);
                    size_t idx = s.nhwc ? ((size_t)y * s.inW + x) * 3 + c : ((size_t)c * s.inH + y) * s.inW + x;
                    store(idx, p);
                }
            }
        }
    };
    if (esz == 1)
        fill([&](size_t i, int p) { dst[i] = lut[p]; });
    else if (esz == 2)
        fill([&](size_t i, int p) { std::memcpy(dst + i * 2, lut + (size_t)p * 2, 2); });
    else
        fill([&](size_t i, int p) { std::memcpy(dst + i * 4, lut + (size_t)p * 4, 4); });

    if (!s.inputRangeLogged)
    {
        double sums[3] = {0.0, 0.0, 0.0};
        float mins[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        float maxs[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        const size_t pixels = (size_t)s.inW * s.inH;
        for (int c = 0; c < 3; c++)
        {
            const float* plane = padded.channel(c);
            for (size_t i = 0; i < pixels; i++)
            {
                const float v = plane[i] / 255.f;
                sums[c] += v;
                mins[c] = std::min(mins[c], v);
                maxs[c] = std::max(maxs[c], v);
            }
        }
        QLOGI("first input RGB: R=[%.4f,%.4f] mean=%.4f G=[%.4f,%.4f] mean=%.4f "
              "B=[%.4f,%.4f] mean=%.4f layout=%s",
              mins[0], maxs[0], sums[0] / pixels, mins[1], maxs[1], sums[1] / pixels,
              mins[2], maxs[2], sums[2] / pixels, s.nhwc ? "NHWC" : "NCHW");
        s.inputRangeLogged = true;
    }

    const auto tPre = std::chrono::steady_clock::now();

    Qnn_ErrorHandle_t err = s.qnn.graphExecute(s.graph, s.inputs.data(), (uint32_t)s.inputs.size(),
                                               s.outputs.data(), (uint32_t)s.outputs.size(), nullptr, nullptr);
    if (err != QNN_SUCCESS)
    {
        QLOGE("graphExecute failed: %lld", (long long)err);
        return -1;
    }

    const auto tExec = std::chrono::steady_clock::now();

    const size_t total = (size_t)s.channels * s.anchors;
    auto readOutput = [&](int outputIndex, int outputChannels, bool outputChannelFirst, int dstChannel) {
        const Qnn_Tensor_t& tensor = s.outputs[outputIndex];
        const Qnn_DataType_t type = tensorType(tensor);
        const Quant quant = quantOf(tensor);
        const uint8_t* src = s.outputBufs[outputIndex].data();
        for (int c = 0; c < outputChannels; c++)
            for (int a = 0; a < s.anchors; a++)
            {
                const size_t srcIndex = outputChannelFirst ? (size_t)c * s.anchors + a
                                                           : (size_t)a * outputChannels + c;
                s.outF[(size_t)(dstChannel + c) * s.anchors + a] = readElement(src, srcIndex, type, quant);
            }
    };
    if (s.splitOutput)
    {
        readOutput(0, 4, s.boxChannelFirst, 0);
        readOutput(1, s.numClass, s.scoreChannelFirst, 4);
    }
    else
    {
        readOutput(0, s.channels, s.channelFirst, 0);
    }

    const float xMul = s.boxesNormalized ? (float)s.inW : 1.f;
    const float yMul = s.boxesNormalized ? (float)s.inH : 1.f;
    auto at = [&](int ch, int a) -> float {
        return s.channelFirst ? s.outF[(size_t)ch * s.anchors + a] : s.outF[(size_t)a * s.channels + ch];
    };

    std::vector<Object>& proposals = s.proposals;
    proposals.clear();
    proposals.reserve(std::min(s.anchors, 300));
    float frameMaxScore = -FLT_MAX;
    for (int a = 0; a < s.anchors; a++)
    {
        int label = 0;
        float score = at(4, a);
        for (int c = 1; c < s.numClass; c++)
        {
            float v = at(4 + c, a);
            if (v > score)
            {
                score = v;
                label = c;
            }
        }
        frameMaxScore = std::max(frameMaxScore, score);
        if (score < prob_threshold) continue;
        float cx = at(0, a) * xMul;
        float cy = at(1, a) * yMul;
        float w = at(2, a) * xMul;
        float h = at(3, a) * yMul;
        Object o;
        o.x = cx - w * 0.5f;
        o.y = cy - h * 0.5f;
        o.w = w;
        o.h = h;
        o.label = label;
        o.prob = score;
        proposals.push_back(o);
    }

    if (!s.outputRangeLogged)
    {
        float minValue = FLT_MAX;
        float maxValue = -FLT_MAX;
        float maxScore = -FLT_MAX;
        float maxBox = -FLT_MAX;
        int bestAnchor = -1;
        for (size_t i = 0; i < total; i++)
        {
            minValue = std::min(minValue, s.outF[i]);
            maxValue = std::max(maxValue, s.outF[i]);
        }
        for (int a = 0; a < s.anchors; a++)
        {
            for (int c = 0; c < 4; c++) maxBox = std::max(maxBox, at(c, a));
            for (int c = 0; c < s.numClass; c++)
            {
                const float score = at(4 + c, a);
                if (score > maxScore)
                {
                    maxScore = score;
                    bestAnchor = a;
                }
            }
        }
        QLOGI("first output: range=[%.6f, %.6f] maxBox=%.6f maxScore=%.6f proposals=%zu threshold=%.4f",
              minValue, maxValue, maxBox, maxScore, proposals.size(), prob_threshold);
        for (int c = 0; c < s.channels; c++)
        {
            float channelMin = FLT_MAX;
            float channelMax = -FLT_MAX;
            for (int a = 0; a < s.anchors; a++)
            {
                const float v = at(c, a);
                channelMin = std::min(channelMin, v);
                channelMax = std::max(channelMax, v);
            }
            QLOGI("output channel %d: range=[%.6f, %.6f]", c, channelMin, channelMax);
        }
        if (bestAnchor >= 0)
        {
            std::string values;
            for (int c = 0; c < s.channels; c++)
            {
                if (!values.empty()) values += ",";
                values += std::to_string(at(c, bestAnchor));
            }
            QLOGI("best anchor %d values=[%s]", bestAnchor, values.c_str());
        }
        s.outputRangeLogged = true;
    }

    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    if (proposals.size() > 300) proposals.resize(300);

    std::vector<Object>& kept = s.kept;
    kept.clear();
    kept.reserve(proposals.size());
    for (const Object& p : proposals)
    {
        bool keep = true;
        for (const Object& k : kept)
        {
            if (iouOf(p, k) > nms_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep) kept.push_back(p);
    }

    for (Object o : kept)
    {
        float x0 = (o.x - padX) / scale;
        float y0 = (o.y - padY) / scale;
        float x1 = (o.x + o.w - padX) / scale;
        float y1 = (o.y + o.h - padY) / scale;
        x0 = std::max(0.f, std::min(x0, (float)(width - 1)));
        y0 = std::max(0.f, std::min(y0, (float)(height - 1)));
        x1 = std::max(0.f, std::min(x1, (float)(width - 1)));
        y1 = std::max(0.f, std::min(y1, (float)(height - 1)));
        o.x = x0;
        o.y = y0;
        o.w = x1 - x0;
        o.h = y1 - y0;
        if (o.w > 1.f && o.h > 1.f) objects.push_back(o);
    }

    const auto tEnd = std::chrono::steady_clock::now();
    auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    s.accPre += ms(tStart, tPre);
    s.accExec += ms(tPre, tExec);
    s.accPost += ms(tExec, tEnd);
    if (++s.accFrames == 90)
    {
        QLOGI("avg over 90 frames: preprocess %.1f ms | NPU execute %.1f ms | postprocess %.1f ms | "
              "maxScore %.6f | proposals %zu | detections %zu | threshold %.4f",
              s.accPre / 90, s.accExec / 90, s.accPost / 90, frameMaxScore,
              proposals.size(), objects.size(), prob_threshold);
        s.accPre = s.accExec = s.accPost = 0;
        s.accFrames = 0;
    }
    return 0;
}

#else

struct QnnYolo::Impl
{
};

QnnYolo::QnnYolo() : impl_(new Impl) {}

QnnYolo::~QnnYolo() = default;

std::string QnnYolo::describe() const { return ""; }

std::string QnnYolo::load(AAssetManager*, const std::string&, const std::string&, int, bool)
{
    return "built without the QNN SDK: set qnn.sdk.dir in local.properties (or QNN_SDK_ROOT) and rebuild";
}

int QnnYolo::detect(const unsigned char*, int, int, std::vector<Object>& objects, float, float)
{
    objects.clear();
    return -1;
}

#endif
