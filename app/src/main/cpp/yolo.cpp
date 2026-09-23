#include "yolo.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>

#include <cpu.h>
#include <layer.h>

static std::string detect_weight_precision(AAssetManager* mgr, const std::string& asset_name)
{
    AAsset* asset = AAssetManager_open(mgr, asset_name.c_str(), AASSET_MODE_STREAMING);
    if (!asset)
        return "UNKNOWN";

    bool has_fp16 = false;
    bool has_bf16 = false;
    bool has_int8 = false;
    uint8_t buffer[64 * 1024];

    for (;;)
    {
        const int bytes_read = AAsset_read(asset, buffer, sizeof(buffer));
        if (bytes_read <= 0)
            break;

        for (int offset = 0; offset + 4 <= bytes_read; offset += 4)
        {
            uint32_t tag;
            std::memcpy(&tag, buffer + offset, sizeof(tag));
            if (tag == 0x01306B47u)
                has_fp16 = true;
            else if (tag == 0x01348B83u)
                has_bf16 = true;
            else if (tag == 0x000D4B38u)
                has_int8 = true;
        }
    }
    AAsset_close(asset);

    const int detected_types = (has_fp16 ? 1 : 0) + (has_bf16 ? 1 : 0) + (has_int8 ? 1 : 0);
    if (detected_types > 1)
        return "MIXED";
    if (has_int8)
        return "INT8";
    if (has_fp16)
        return "FP16";
    if (has_bf16)
        return "BF16";
    return "FP32";
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float intersection_area(const Object& a, const Object& b)
{
    float x0 = std::max(a.x, b.x);
    float y0 = std::max(a.y, b.y);
    float x1 = std::min(a.x + a.w, b.x + b.w);
    float y1 = std::min(a.y + a.h, b.y + b.h);
    float iw = std::max(0.f, x1 - x0);
    float ih = std::max(0.f, y1 - y0);
    return iw * ih;
}

static void qsort_descent_inplace(std::vector<Object>& objs, int left, int right)
{
    int i = left, j = right;
    float p = objs[(left + right) / 2].prob;
    while (i <= j)
    {
        while (objs[i].prob > p) i++;
        while (objs[j].prob < p) j--;
        if (i <= j)
        {
            std::swap(objs[i], objs[j]);
            i++;
            j--;
        }
    }
    if (left < j) qsort_descent_inplace(objs, left, j);
    if (i < right) qsort_descent_inplace(objs, i, right);
}

static void qsort_descent_inplace(std::vector<Object>& objs)
{
    if (!objs.empty())
        qsort_descent_inplace(objs, 0, (int)objs.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objs, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = (int)objs.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = objs[i].w * objs[i].h;

    for (int i = 0; i < n; i++)
    {
        const Object& a = objs[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objs[picked[j]];
            float inter = intersection_area(a, b);
            float uni = areas[i] + areas[picked[j]] - inter;
            if (uni > 0.f && inter / uni > nms_threshold)
            {
                keep = 0;
                break;
            }
        }
        if (keep)
            picked.push_back(i);
    }
}

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(int target_w, int target_h,
                                      const std::vector<int>& strides,
                                      std::vector<GridAndStride>& grid_strides)
{
    for (int stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
            for (int g0 = 0; g0 < num_grid_w; g0++)
                grid_strides.push_back({g0, g1, stride});
    }
}

static void generate_proposals_dfl(const std::vector<GridAndStride>& grid_strides,
                                   const ncnn::Mat& pred, int num_class,
                                   float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = (int)grid_strides.size();
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            if (scores[k] > score)
            {
                label = k;
                score = scores[k];
            }
        }

        float box_prob = sigmoid(score);
        if (box_prob < prob_threshold)
            continue;

        ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
        {
            ncnn::Layer* softmax = ncnn::create_layer("Softmax");
            ncnn::ParamDict pd;
            pd.set(0, 1);
            pd.set(1, 1);
            softmax->load_param(pd);
            ncnn::Option opt;
            opt.num_threads = 1;
            opt.use_packing_layout = false;
            softmax->create_pipeline(opt);
            softmax->forward_inplace(bbox_pred, opt);
            softmax->destroy_pipeline(opt);
            delete softmax;
        }

        float pred_ltrb[4];
        for (int k = 0; k < 4; k++)
        {
            float dis = 0.f;
            const float* dis_after_sm = bbox_pred.row(k);
            for (int l = 0; l < reg_max_1; l++)
                dis += l * dis_after_sm[l];
            pred_ltrb[k] = dis * grid_strides[i].stride;
        }

        float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
        float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

        Object obj;
        obj.x = pb_cx - pred_ltrb[0];
        obj.y = pb_cy - pred_ltrb[1];
        obj.w = (pb_cx + pred_ltrb[2]) - obj.x;
        obj.h = (pb_cy + pred_ltrb[3]) - obj.y;
        obj.label = label;
        obj.prob = box_prob;
        objects.push_back(obj);
    }
}

static void generate_proposals_decoded(const ncnn::Mat& pred, int num_class,
                                       float prob_threshold, std::vector<Object>& objects)
{
    const int num_anchors = pred.w;

    for (int i = 0; i < num_anchors; i++)
    {
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float s = pred.row(4 + k)[i];
            if (s > score)
            {
                label = k;
                score = s;
            }
        }
        if (score < prob_threshold)
            continue;

        float cx = pred.row(0)[i];
        float cy = pred.row(1)[i];
        float bw = pred.row(2)[i];
        float bh = pred.row(3)[i];

        Object obj;
        obj.x = cx - bw * 0.5f;
        obj.y = cy - bh * 0.5f;
        obj.w = bw;
        obj.h = bh;
        obj.label = label;
        obj.prob = score;
        objects.push_back(obj);
    }
}

Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolo::load(AAssetManager* mgr, const ModelSpec& _spec, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    spec = _spec;
    precision = detect_weight_precision(mgr, spec.bin_asset);

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();
#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu && (ncnn::get_gpu_count() > 0);
#endif
    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    gpu_loaded = yolo.opt.use_vulkan_compute;

    if (yolo.load_param(mgr, spec.param_asset.c_str()) != 0) return -1;
    if (yolo.load_model(mgr, spec.bin_asset.c_str()) != 0) return -1;

    return 0;
}

int Yolo::detect(const unsigned char* rgba, int width, int height,
                 std::vector<Object>& objects,
                 float prob_threshold, float nms_threshold)
{
    const int target_size = spec.target_size;

    int w = width;
    int h = height;
    float scale;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = (int)roundf(h * scale);
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = (int)roundf(w * scale);
    }

    int pixel_type = spec.bgr ? ncnn::Mat::PIXEL_RGBA2BGR : ncnn::Mat::PIXEL_RGBA2RGB;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        rgba, pixel_type, width, height, w, h);

    int wpad, hpad;
    if (spec.decoded)
    {
        wpad = target_size - w;
        hpad = target_size - h;
    }
    else
    {
        wpad = (w + 31) / 32 * 32 - w;
        hpad = (h + 31) / 32 * 32 - h;
    }
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad,
                           hpad / 2, hpad - hpad / 2,
                           wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, spec.decoded ? 114.f : 0.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(nullptr, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();
    ex.input(spec.input_name.c_str(), in_pad);

    ncnn::Mat out;
    ex.extract(spec.output_name.c_str(), out);

    std::vector<Object> proposals;
    if (spec.decoded)
    {
        generate_proposals_decoded(out, spec.num_class, prob_threshold, proposals);
    }
    else
    {
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
        generate_proposals_dfl(grid_strides, out, spec.num_class, prob_threshold, proposals);
    }

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    objects.clear();
    objects.reserve(picked.size());
    for (int idx : picked)
    {
        Object o = proposals[idx];
        float x0 = (o.x - wpad / 2) / scale;
        float y0 = (o.y - hpad / 2) / scale;
        float x1 = (o.x + o.w - wpad / 2) / scale;
        float y1 = (o.y + o.h - hpad / 2) / scale;

        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        o.x = x0;
        o.y = y0;
        o.w = x1 - x0;
        o.h = y1 - y0;
        if (o.w > 1.f && o.h > 1.f)
            objects.push_back(o);
    }

    return 0;
}
