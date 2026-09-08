#ifndef YOLOVULKAN_YOLO_H
#define YOLOVULKAN_YOLO_H

#include <string>
#include <vector>

#include <android/asset_manager.h>
#include <net.h>

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

struct ModelSpec
{
    std::string param_asset;
    std::string bin_asset;
    std::string input_name;
    std::string output_name;
    int target_size = 640;
    int num_class = 80;
    bool decoded = false;
    bool bgr = false;
};

class Yolo
{
public:
    Yolo();

    int load(AAssetManager* mgr, const ModelSpec& spec, bool use_gpu);

    int detect(const unsigned char* rgba, int width, int height,
               std::vector<Object>& objects,
               float prob_threshold = 0.25f, float nms_threshold = 0.45f);

    bool has_gpu() const { return gpu_loaded; }

private:
    ncnn::Net yolo;
    ModelSpec spec;
    bool gpu_loaded = false;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif
