#ifndef YOLOVULKAN_QNN_YOLO_H
#define YOLOVULKAN_QNN_YOLO_H

#include <memory>
#include <string>
#include <vector>

#include <android/asset_manager.h>

#include "yolo.h"

class QnnYolo
{
public:
    QnnYolo();
    ~QnnYolo();

    std::string load(AAssetManager* mgr, const std::string& model_asset,
                     const std::string& native_lib_dir, int num_class, bool boxes_normalized);

    std::string describe() const;

    int detect(const unsigned char* rgba, int width, int height,
               std::vector<Object>& objects,
               float prob_threshold, float nms_threshold);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
