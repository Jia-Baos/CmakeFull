#pragma once
#include "../../model_det.hpp"

#include "Interpreter.hpp"

#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

typedef struct HeadInfo_ {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

typedef struct BoxInfo_ {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef struct CenterPrior_ {
    int x;
    int y;
    int stride;
} CenterPrior;

class NanoDetMNN : public DetModel {
public:
    NanoDetMNN(const std::string &config_path);
    virtual ~NanoDetMNN();
    virtual std::shared_ptr<DetOutput> Detect(const cv::Mat &img);
    static std::shared_ptr<DetModel> GetModel(const std::string &config_path);

    void decode_infer(MNN::Tensor *pred, std::vector<CenterPrior> &center_priors, float threshold, std::vector<std::vector<BoxInfo>> &results);
    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride);
    void nms(std::vector<BoxInfo> &input_boxes, float nms_thresh);

    // modify these parameters to the same with your config if you want to use your own model
    int input_size[2] = { 416, 416 };             // input height and width
    int num_class = 80;                           // number of classes. 80 for COCO
    int reg_max = 7;                              // `reg_max` set in the training config. Default: 7.
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.

    std::string input_name = "data";
    std::string output_name = "output";

    std::vector<BoxInfo> m_result_list;

private:
    std::shared_ptr<MNN::Interpreter> NanoDet_interpreter;
    MNN::Session *NanoDet_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;

    int m_thread_num;
    int m_class_num;

    float m_nms_threshold;
    float m_confidence_threshold;

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
};

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template <typename _Tp>
inline int activation_function_softmax(const _Tp *src, _Tp *dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}